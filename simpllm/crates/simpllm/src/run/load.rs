use gpt2weights::{HParams, ModelPath, ModelShape, NormOffsets, Offsets, load_metadata};
use simpllm_core::llm::ModelLoader;
use simpllm_core::tensor::{LayerNorm, Tensor, TensorBackend};
use simpllm_core::transformer::{Attention, Ffn, TransformerBlock};
use std::error::Error;
use std::fs;

pub fn load_model<B: TensorBackend>(path: &ModelPath) -> Result<(ModelLoader<B>, ModelShape), Box<dyn Error>> {
    let shape = load_metadata(path)?;
    let h_params = &shape.h_params;

    let tensor_data = fs::read(path.path(&shape.file_names.tensors))?;
    let offsets = &shape.tensor_offsets;

    // metadata.tok_embed is [n_vocab, dim]. Pos is [n_seq, dim]
    eprint!("loading token embeddings... ");

    let tok_embed = load_tensor::<B, _>([h_params.vocab_size, h_params.n_embd], &tensor_data, offsets.tok_embed)?;
    eprint!("position embeddings... ");
    let pos_embed = load_tensor::<B, _>([h_params.n_ctx, h_params.n_embd], &tensor_data, offsets.pos_embed)?;

    let mut layers = Vec::with_capacity(h_params.n_layer);
    eprint!("{} layers [ ", h_params.n_layer);
    for layer_idx in 0..h_params.n_layer {
        eprint!("{} ", layer_idx + 1);
        let transformer = load_transformer(&tensor_data, layer_idx, &shape, &h_params)?;
        layers.push(transformer);
    }

    eprint!("] ... final normalization...");
    let final_norm = load_norm::<B>(&tensor_data, offsets.final_norm, h_params.n_embd)?;
    eprintln!("Model loaded!");

    let loader = ModelLoader {
        tok_embed,
        pos_embed,
        layers,
        final_norm,
        eos_token: h_params.eos_token_id,
    };
    Ok((loader, shape))
}

fn load_norm<B: TensorBackend>(data: &[u8], norm: NormOffsets, n_embed: usize) -> Result<B::LayerNorm, Box<dyn Error>> {
    let scale = load_tensor::<B, _>([n_embed], data, norm.scale)?;
    let bias = load_tensor::<B, _>([n_embed], data, norm.bias)?;
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
    data: &[u8],
    layer_idx: usize,
    model_shape: &ModelShape,
    h_params: &HParams,
) -> Result<TransformerBlock<B>, Box<dyn Error>> {
    // QKV should ba a tensor of [d x 3d]
    let (d, qkv_3d) = (h_params.n_embd, h_params.n_embd * 3);
    let offsets = &model_shape.tensor_offsets.transformers[layer_idx];

    // attention

    let mut attn = Attention::<B>::new(d, h_params.n_head);

    let qkv_weights = load_tensor::<B, _>([d, qkv_3d], data, offsets.attn_qkv.weight)?;
    let qkv_bias = load_tensor::<B, _>([qkv_3d], data, offsets.attn_qkv.bias)?;
    attn.qkv_mut().set(&qkv_weights, &qkv_bias);

    let attn_wo = load_tensor::<B, _>([d, d], data, offsets.attn_wo.weight)?;
    let attn_o_bias = load_tensor::<B, _>([d], data, offsets.attn_wo.bias)?;
    attn.o_mut().set(&attn_wo, &attn_o_bias);

    let attn_norm = load_norm::<B>(data, offsets.before_attn_norm, d)?;

    // ffn
    let hidden_d = model_shape.transformer[layer_idx].ffn_hidden_layer_embed;
    let mut ffn = Ffn::<B>::new(d, &[hidden_d], d);

    let ffn_hidden_weights = load_tensor::<B, _>([d, hidden_d], data, offsets.ffn_hidden.weight)?;
    let ffn_hidden_bias = load_tensor::<B, _>([hidden_d], data, offsets.ffn_hidden.bias)?;
    let ffn_output_weights = load_tensor::<B, _>([hidden_d, d], data, offsets.ffn_output.weight)?;
    let ffn_output_bias = load_tensor::<B, _>([d], data, offsets.ffn_output.bias)?;
    let ffn_norm = load_norm::<B>(data, offsets.before_ffn_norm, d)?;

    ffn.layer_mut(0).set(&ffn_hidden_weights, &ffn_hidden_bias);
    ffn.layer_mut(1).set(&ffn_output_weights, &ffn_output_bias);

    let block = TransformerBlock::new(attn_norm, attn, ffn_norm, ffn);

    Ok(block)
}

fn load_tensor<B: TensorBackend, const R: usize>(
    size: [usize; R],
    data: &[u8],
    offsets: Offsets,
) -> Result<B::Tensor<R>, Box<dyn Error>> {
    let floats = load_floats(data, offsets)?;
    let tensor = populate_tensor::<B, _>(size, &floats);
    Ok(tensor)
}

fn populate_tensor<B: TensorBackend, const R: usize>(size: [usize; R], vals: &[f32]) -> B::Tensor<R> {
    let mut t = B::Tensor::zeros(size);
    t.reset_values(vals);
    t
}

fn load_floats(data: &[u8], offsets: Offsets) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(data[offsets.0..offsets.1]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}
