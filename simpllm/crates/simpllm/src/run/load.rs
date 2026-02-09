use crate::llm::ModelLoader;
use crate::tensor::{LayerNorm, Shape, Tensor, TensorBackend};
use crate::transformer::{Attention, Ffn, TransformerBlock};
use gpt2weights::{
    HParams, ModelFile, ModelPath, NormFile, NormVariant, TransformerN, TransformerShape, load_metadata, read_nicely,
};
use std::error::Error;
use std::io::Read;

pub fn load_model<B: TensorBackend>(path: &ModelPath) -> Result<ModelLoader<B>, Box<dyn Error>> {
    let (shape, h_params) = load_metadata(path)?;

    // metadata.tok_embed is [n_vocab, dim]. Pos is [n_seq, dim]
    eprint!("loading token embeddings... ");

    let tok_embed = load_tensor::<B, _>([h_params.vocab_size, h_params.n_embd], path, ModelFile::TokEmbed)?;
    eprint!("position embeddings... ");
    let pos_embed = load_tensor::<B, _>([h_params.n_ctx, h_params.n_embd], path, ModelFile::PosEmbed)?;

    let mut layers = Vec::with_capacity(h_params.n_layer);
    eprint!("{} layers [ ", h_params.n_layer);
    for layer_idx in 0..h_params.n_layer {
        let layer_meta = &shape.transformer[layer_idx];
        eprint!("{} ", layer_idx + 1);
        let transformer = load_transformer(path, layer_idx, &layer_meta, &h_params)?;
        layers.push(transformer);
    }

    eprint!("] ... final normalization...");
    let final_norm = load_norm::<B>(&path, NormFile::Final, h_params.n_embd)?;
    eprintln!("Model loaded!");

    Ok(ModelLoader {
        tok_embed,
        pos_embed,
        layers,
        final_norm,
    })
}

fn load_norm<B: TensorBackend>(
    path: &ModelPath,
    norm: NormFile,
    n_embed: usize,
) -> Result<B::LayerNorm, Box<dyn Error>> {
    let scale = load_tensor::<B, _>([n_embed], path, ModelFile::Norm(norm, NormVariant::Scale))?;
    let bias = load_tensor::<B, _>([n_embed], path, ModelFile::Norm(norm, NormVariant::Bias))?;
    let norm = B::LayerNorm::new(scale, bias, 1e-5);
    Ok(norm)
}

/// Loads the files for one transformer block.
///
/// The files are something like:
///
/// - transformer.00.attn.output.bias.bin
/// - transformer.00.attn.output.weights.bin
/// - transformer.00.attn.qkv.bias.bin
/// - transformer.00.attn.qkv.weights.bin
/// - transformer.00.attn_norm.bias.bin
/// - transformer.00.attn_norm.scale.bin
/// - transformer.00.ffn.hidden.bias.bin
/// - transformer.00.ffn.hidden.weights.bin
/// - transformer.00.ffn.output.bias.bin
/// - transformer.00.ffn.output.weights.bin
/// - transformer.00.ffn_norm.bias.bin
/// - transformer.00.ffn_norm.scale.bin
fn load_transformer<B: TensorBackend>(
    path: &ModelPath,
    layer_idx: usize,
    metadata: &TransformerShape,
    h_params: &HParams,
) -> Result<TransformerBlock<B>, Box<dyn Error>> {
    use ModelFile::Transformer;
    use gpt2weights::TransformerComponent::*;
    use gpt2weights::WeightVariant::*;

    // QKV should ba a tensor of [d x 3d]
    let (d, qkv_3d) = (h_params.n_embd, h_params.n_embd * 3);

    let h = TransformerN(layer_idx);

    // attention

    let mut attn = Attention::<B>::new(d, h_params.n_head);

    let qkv_weights = load_tensor::<B, _>([d, qkv_3d], path, Transformer(h, Qkv, Weights))?;
    let qkv_bias = load_tensor::<B, _>([qkv_3d], path, Transformer(h, Qkv, Bias))?;
    attn.qkv_mut().set(&qkv_weights, &qkv_bias);

    let attn_wo = load_tensor::<B, _>([d, d], path, Transformer(h, AttnOutput, Weights))?;
    let attn_o_bias = load_tensor::<B, _>([d], path, Transformer(h, AttnOutput, Bias))?;
    attn.o_mut().set(&attn_wo, &attn_o_bias);

    let attn_norm = load_norm::<B>(path, NormFile::BeforeAttention(h), d)?;

    // ffn
    let hidden_d = metadata.ffn_hidden_layer_embed;
    let mut ffn = Ffn::<B>::new(d, &[hidden_d], d);

    let ffn_hidden_weights = load_tensor::<B, _>([d, hidden_d], path, Transformer(h, FfnHidden, Weights))?;
    let ffn_hidden_bias = load_tensor::<B, _>([hidden_d], path, Transformer(h, FfnHidden, Bias))?;
    let ffn_output_weights = load_tensor::<B, _>([hidden_d, d], path, Transformer(h, FfnOutput, Weights))?;
    let ffn_output_bias = load_tensor::<B, _>([d], path, Transformer(h, FfnOutput, Bias))?;
    let ffn_norm = load_norm::<B>(path, NormFile::BeforeFfn(h), d)?;

    ffn.layer_mut(0).set(&ffn_hidden_weights, &ffn_hidden_bias);
    ffn.layer_mut(1).set(&ffn_output_weights, &ffn_output_bias);

    let block = TransformerBlock::new(attn_norm, attn, ffn_norm, ffn);

    Ok(block)
}

fn load_tensor<B: TensorBackend, const R: usize>(
    size: [usize; R],
    path: &ModelPath,
    file: ModelFile,
) -> Result<B::Tensor<R>, Box<dyn Error>> {
    let floats = load_floats(size, path, file)?;
    let tensor = populate_tensor::<B, _>(size, &floats);
    Ok(tensor)
}

fn populate_tensor<B: TensorBackend, const R: usize>(size: [usize; R], vals: &[f32]) -> B::Tensor<R> {
    let mut t = B::Tensor::new(size);
    t.reset_values(vals);
    t
}

fn load_floats<const R: usize>(
    size: [usize; R],
    path: &ModelPath,
    file: ModelFile,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let size: Shape<R> = size.into();

    let expected_size = size.num_elements() * size_of::<f32>();
    let mut contents_u8: Vec<u8> = vec![0; expected_size];
    let mut contents = read_nicely(path, file)?;
    contents.read_exact(&mut contents_u8)?;

    let extra_bytes_count = std::io::copy(&mut contents, &mut std::io::sink())?;

    if extra_bytes_count > 0 {
        let multiplier = (extra_bytes_count as f64) / (expected_size as f64);
        let s = format!(
            "got extra bytes in {path:?}. expected {expected_size} (for shape {size}), saw {extra_bytes_count} extra ({multiplier:.2}x)"
        );
        return Err(s.into());
    }

    Ok(contents_u8
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}
