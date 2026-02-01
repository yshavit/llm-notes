use crate::llm::ModelLoader;
use crate::load::load_metadata;
use crate::load::metadata::{NormShape, TransformerShape};
use crate::load::path::{ModelPath, read_nicely};
use crate::tensor::{Shape, Tensor};
use crate::transformer::{Attention, Ffn, Norm, TransformerBlock};
use std::fs;
use std::io::Read;
use std::path::PathBuf;

pub fn load_model(path: &ModelPath) -> super::Result<ModelLoader> {
    let (shape, h_params) = load_metadata(path)?;

    // metadata.tok_embed is [n_vocab, dim]. Pos is [n_seq, dim]
    eprint!("loading token embeddings... ");
    let tok_embed = load_tensor([shape.tok_embed[0], shape.tok_embed[1]], &path.path("tok_embed.bin"))?;
    eprint!("position embeddings... ");
    let pos_embed = load_tensor([shape.pos_embed[0], shape.pos_embed[1]], &path.path("pos_embed.bin"))?;

    let mut layers = Vec::with_capacity(shape.layer.len());
    eprint!("{} layers [ ", shape.layer.len());
    for (layer_idx, layer_meta) in shape.layer.iter().enumerate() {
        eprint!("{} ", layer_idx + 1);
        let transformer = load_transformer(path, layer_idx, layer_meta, h_params.n_head)?;
        layers.push(transformer);
    }

    eprint!("] ... final normalization...");
    let final_norm = load_norm(&path, "final_norm", &shape.final_norm)?;
    eprintln!("Model loaded!");

    Ok(ModelLoader {
        tok_embed,
        pos_embed,
        layers,
        final_norm,
    })
}

fn load_norm(path: &ModelPath, prefix: &str, metadata: &NormShape) -> super::Result<Norm> {
    assert_eq!(metadata.scale, metadata.bias);
    let scale = load_tensor([metadata.scale], &path.path([prefix, ".scale.bin"]))?;
    let bias = load_tensor([metadata.bias], &path.path([prefix, ".bias.bin"]))?;
    let mut norm = Norm::new(metadata.scale);
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
    n_heads: usize,
) -> super::Result<TransformerBlock> {
    let layer_idx_string = find_transformer_prefix(path, layer_idx)?;

    // QKV should ba a tensor of [d x 3d]
    let (d, qkv_3d) = metadata.attn.qkv.weights;
    assert_eq!(qkv_3d, d * 3);
    assert_eq!(metadata.attn.qkv.bias, d * 3);

    let t_path = |desc| path.path(["transformer.", &layer_idx_string, ".", desc, ".bin"]);

    // attention

    let mut attn = Attention::new(d, n_heads);

    let qkv_floats = load_floats([1, d, qkv_3d], &t_path("attn.qkv.weights"))?;
    let mut wq = Vec::with_capacity(d * d);
    let mut wk = Vec::with_capacity(d * d);
    let mut wv = Vec::with_capacity(d * d);

    for row in 0..d {
        let row_start = row * qkv_3d; // qkv_3d = 3*d
        wq.extend_from_slice(&qkv_floats[row_start..row_start + d]);
        wk.extend_from_slice(&qkv_floats[row_start + d..row_start + 2 * d]);
        wv.extend_from_slice(&qkv_floats[row_start + 2 * d..row_start + 3 * d]);
    }
    let attn_wq = populate_tensor([d, d], &wq);
    let attn_wk = populate_tensor([d, d], &wk);
    let attn_wv = populate_tensor([d, d], &wv);

    let qkv_bias_floats = load_floats([metadata.attn.qkv.bias], &t_path("attn.qkv.bias"))?;
    let mut qkv_bias_chunks = qkv_bias_floats.chunks_exact(d);
    let attn_q_bias = populate_tensor([d], &qkv_bias_chunks.next().unwrap());
    let attn_k_bias = populate_tensor([d], &qkv_bias_chunks.next().unwrap());
    let attn_v_bias = populate_tensor([d], &qkv_bias_chunks.next().unwrap());
    assert_eq!(qkv_bias_chunks.count(), 0);

    let attn_wo = load_tensor([d, d], &t_path("attn.output.weights"))?;
    let attn_o_bias = load_tensor([d], &t_path("attn.output.bias"))?;

    attn.q_mut().set(&attn_wq, &attn_q_bias);
    attn.k_mut().set(&attn_wk, &attn_k_bias);
    attn.v_mut().set(&attn_wv, &attn_v_bias);
    attn.o_mut().set(&attn_wo, &attn_o_bias);

    let attn_norm = load_norm(
        path,
        &format!("transformer.{layer_idx_string}.attn_norm"),
        &metadata.attn_norm,
    )?;

    // ffn
    let ffn_m = &metadata.ffn;
    let mut ffn = Ffn::new(d, &[ffn_m.hidden.bias], d);

    let ffn_hidden_weights = load_tensor(
        [ffn_m.hidden.weights.0, ffn_m.hidden.weights.1],
        &t_path("ffn.hidden.weights"),
    )?;
    let ffn_hidden_bias = load_tensor([ffn_m.hidden.bias], &t_path("ffn.hidden.bias"))?;
    let ffn_output_weights = load_tensor(
        [ffn_m.output.weights.0, ffn_m.output.weights.1],
        &t_path("ffn.output.weights"),
    )?;
    let ffn_output_bias = load_tensor([ffn_m.output.bias], &t_path("ffn.output.bias"))?;
    let ffn_norm = load_norm(
        path,
        &format!("transformer.{layer_idx_string}.ffn_norm"),
        &metadata.attn_norm,
    )?;

    ffn.layer_mut(0).set(&ffn_hidden_weights, &ffn_hidden_bias);
    ffn.layer_mut(1).set(&ffn_output_weights, &ffn_output_bias);

    let block = TransformerBlock::new(attn_norm, attn, ffn_norm, ffn);

    Ok(block)
}

fn find_transformer_prefix(path: &ModelPath, layer_idx: usize) -> super::Result<String> {
    for padding_len in 0..10 {
        let num = format!("{}{layer_idx}", "0".repeat(padding_len));
        // look for the attention qkv weights, arbitrarily; any of them should be okay
        let candidate = path.path(["transformer.", &num, ".attn.qkv.weights.bin"]);
        if fs::exists(candidate)? {
            return Ok(num);
        }
    }
    Err(format!("couldn't find files for layer {layer_idx}").into())
}

fn load_tensor<const R: usize>(size: [usize; R], path: &PathBuf) -> super::Result<Tensor<R>> {
    let floats = load_floats(size, &path)?;
    Ok(populate_tensor(size, &floats))
}

fn populate_tensor<const R: usize>(size: [usize; R], vals: &[f32]) -> Tensor<R> {
    let mut t = Tensor::new(size);
    t.reset_values(vals);
    t
}

fn load_floats<const R: usize>(size: [usize; R], path: &PathBuf) -> super::Result<Vec<f32>> {
    let size: Shape<R> = size.into();

    let expected_size = size.num_elements() * size_of::<f32>();
    let mut contents_u8: Vec<u8> = vec![0; expected_size];
    let mut contents = read_nicely(path)?;
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
