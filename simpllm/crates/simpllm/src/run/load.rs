use crate::cputensor::{Shape, Tensor};
use crate::llm::ModelLoader;
use crate::transformer::{Attention, Ffn, Norm, TransformerBlock};
use gpt2weights::{
    HParams, ModelFile, ModelPath, NormFile, NormVariant, TransformerN, TransformerShape, load_metadata, read_nicely,
};
use std::error::Error;
use std::io::Read;

pub fn load_model(path: &ModelPath) -> Result<ModelLoader, Box<dyn Error>> {
    let (shape, h_params) = load_metadata(path)?;

    // metadata.tok_embed is [n_vocab, dim]. Pos is [n_seq, dim]
    eprint!("loading token embeddings... ");

    let tok_embed = load_tensor([h_params.vocab_size, h_params.n_embd], path, ModelFile::TokEmbed)?;
    eprint!("position embeddings... ");
    let pos_embed = load_tensor([h_params.n_ctx, h_params.n_embd], path, ModelFile::PosEmbed)?;

    let mut layers = Vec::with_capacity(h_params.n_layer);
    eprint!("{} layers [ ", h_params.n_layer);
    for layer_idx in 0..h_params.n_layer {
        let layer_meta = &shape.transformer[layer_idx];
        eprint!("{} ", layer_idx + 1);
        let transformer = load_transformer(path, layer_idx, &layer_meta, &h_params)?;
        layers.push(transformer);
    }

    eprint!("] ... final normalization...");
    let final_norm = load_norm(&path, NormFile::Final, h_params.n_embd)?;
    eprintln!("Model loaded!");

    Ok(ModelLoader {
        tok_embed,
        pos_embed,
        layers,
        final_norm,
    })
}

fn load_norm(path: &ModelPath, norm: NormFile, n_embed: usize) -> Result<Norm, Box<dyn Error>> {
    let scale = load_tensor([n_embed], path, ModelFile::Norm(norm, NormVariant::Scale))?;
    let bias = load_tensor([n_embed], path, ModelFile::Norm(norm, NormVariant::Bias))?;
    let mut norm = Norm::new(n_embed);
    norm.set(&scale, &bias);
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
fn load_transformer(
    path: &ModelPath,
    layer_idx: usize,
    metadata: &TransformerShape,
    h_params: &HParams,
) -> Result<TransformerBlock, Box<dyn Error>> {
    use ModelFile::Transformer;
    use gpt2weights::TransformerComponent::*;
    use gpt2weights::WeightVariant::*;

    // QKV should ba a tensor of [d x 3d]
    let (d, qkv_3d) = (h_params.n_embd, h_params.n_embd * 3);

    let h = TransformerN(layer_idx);

    // attention

    let mut attn = Attention::new(d, h_params.n_head);

    let qkv_weights = load_tensor([d, qkv_3d], path, Transformer(h, Qkv, Weights))?;
    let qkv_bias = load_tensor([qkv_3d], path, Transformer(h, Qkv, Bias))?;
    attn.qkv_mut().set(&qkv_weights, &qkv_bias);

    let attn_wo = load_tensor([d, d], path, Transformer(h, AttnOutput, Weights))?;
    let attn_o_bias = load_tensor([d], path, Transformer(h, AttnOutput, Bias))?;
    attn.o_mut().set(&attn_wo, &attn_o_bias);

    let attn_norm = load_norm(path, NormFile::BeforeAttention(h), d)?;

    // ffn
    let hidden_d = metadata.ffn_hidden_layer_embed;
    let mut ffn = Ffn::new(d, &[hidden_d], d);

    let ffn_hidden_weights = load_tensor([d, hidden_d], path, Transformer(h, FfnHidden, Weights))?;
    let ffn_hidden_bias = load_tensor([hidden_d], path, Transformer(h, FfnHidden, Bias))?;
    let ffn_output_weights = load_tensor([hidden_d, d], path, Transformer(h, FfnOutput, Weights))?;
    let ffn_output_bias = load_tensor([d], path, Transformer(h, FfnOutput, Bias))?;
    let ffn_norm = load_norm(path, NormFile::BeforeFfn(h), d)?;

    ffn.layer_mut(0).set(&ffn_hidden_weights, &ffn_hidden_bias);
    ffn.layer_mut(1).set(&ffn_output_weights, &ffn_output_bias);

    let block = TransformerBlock::new(attn_norm, attn, ffn_norm, ffn);

    Ok(block)
}

fn load_tensor<const R: usize>(
    size: [usize; R],
    path: &ModelPath,
    file: ModelFile,
) -> Result<Tensor<R>, Box<dyn Error>> {
    let floats = load_floats(size, path, file)?;
    Ok(populate_tensor(size, &floats))
}

fn populate_tensor<const R: usize>(size: [usize; R], vals: &[f32]) -> Tensor<R> {
    let mut t = Tensor::new(size);
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
