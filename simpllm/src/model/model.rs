use crate::model::load_metadata;
use crate::model::metadata::{Norm, Transformer};
use crate::model::path::ModelPath;
use crate::tensor::{Matrix, Shape, Tensor, Vector};
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

pub struct Model {
    tok_embed: Vector,
    pos_embed: Vector,
    final_norm: NormTensors,
    layers: Vec<TransformerData>,
}

impl Model {
    pub fn load(path: &ModelPath) -> super::Result<Self> {
        let metadata = load_metadata(path)?;

        // metadata.tok_embed is [vocab, dim]. We just care about dim. Ditto for pos_embed.
        let tok_embed = load_tensor([metadata.tok_embed[1]], &path.path("tok_embed.bin"))?;
        let pos_embed = load_tensor([metadata.pos_embed[1]], &path.path("pos_embed.bin"))?;

        let final_norm = NormTensors::load(&path, "final_norm", &metadata.final_norm)?;

        let mut layers = Vec::with_capacity(metadata.layer.len());
        for (layer_idx, layer_meta) in metadata.layer.iter().enumerate() {
            let layer = TransformerData::load(path, layer_idx, layer_meta)?;
            layers.push(layer);
        }

        Ok(Self {
            tok_embed,
            pos_embed,
            final_norm,
            layers,
        })
    }
}

struct NormTensors {
    bias: Vector,
    scale: Vector,
}

impl NormTensors {
    fn load(path: &ModelPath, prefix: &str, metadata: &Norm) -> super::Result<Self> {
        let scale = load_tensor([metadata.bias], &path.path([prefix, ".scale.bin"]))?;
        let bias = load_tensor([metadata.bias], &path.path([prefix, ".scale.bin"]))?;
        Ok(Self { scale, bias })
    }
}

struct WeightData {
    weight: Matrix,
    bias: Vector,
}

struct TransformerData {
    attn_norm: NormTensors,
    attn_wq: WeightData,
    attn_wk: WeightData,
    attn_wv: WeightData,
    attn_wo: WeightData,

    ffn_norm: NormTensors,
    ffn_hidden: WeightData,
    ffn_output: WeightData,
}

impl TransformerData {
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
    fn load(path: &ModelPath, layer_idx: usize, metadata: &Transformer) -> super::Result<Self> {
        let layer_idx_string = Self::find_prefix(path, layer_idx)?;

        let qkv_weights = &metadata.attn.qkv.weights;
        // QKV should ba a tensor of [d x 3d]
        let (d, qkv_3d) = metadata.attn.qkv.weights;
        assert_eq!(qkv_3d, d * 3);
        assert_eq!(metadata.attn.qkv.bias, d * 3);

        let t_path = |desc| path.path(["transformer.", &layer_idx_string, desc, ".bin"]);

        // attention

        let qkv_floats = load_floats([1, d, qkv_3d], &t_path("attn.qkv.weights"))?;
        let mut qkv_chunks = qkv_floats.chunks_exact(d);
        let attn_wq = populate_tensor([d, d], &qkv_chunks.next().unwrap());
        let attn_wk = populate_tensor([d, d], &qkv_chunks.next().unwrap());
        let attn_wv = populate_tensor([d, d], &qkv_chunks.next().unwrap());
        assert_eq!(qkv_chunks.count(), 0);

        let qkv_bias_floats = load_floats([metadata.attn.qkv.bias], &t_path("attn.qkv.bias"))?;
        let mut qkv_bias_chunks = qkv_floats.chunks_exact(d);
        let attn_q_bias = populate_tensor([d], &qkv_bias_chunks.next().unwrap());
        let attn_k_bias = populate_tensor([d], &qkv_bias_chunks.next().unwrap());
        let attn_v_bias = populate_tensor([d], &qkv_bias_chunks.next().unwrap());
        assert_eq!(qkv_bias_chunks.count(), 0);

        let attn_wo = load_tensor([d, d], &t_path("attn_output_weights"))?;
        let attn_o_bias = load_tensor([d], &t_path("attn_output_bias"))?;

        let attn_norm = NormTensors::load(
            path,
            &format!("transformer.{layer_idx_string}.attn_norm"),
            &metadata.attn_norm,
        )?;

        // ffn
        let ffn_m = &metadata.ffn;
        let ffn_hidden_weights = load_tensor(
            [ffn_m.hidden.weights.0, ffn_m.hidden.weights.1],
            &t_path("ffn.hidden.weights"),
        )?;
        let ffn_hidden_bias = load_tensor([ffn_m.hidden.bias], &t_path("ffn.hidden.bias"))?;
        let ffn_output_weights = load_tensor(
            [ffn_m.output.weights.0, ffn_m.output.weights.1],
            &t_path("ffn.hidden.weights"),
        )?;
        let ffn_output_bias = load_tensor([ffn_m.output.bias], &t_path("ffn.hidden.bias"))?;
        let ffn_norm = NormTensors::load(
            path,
            &format!("transformer.{layer_idx_string}.attn_norm"),
            &metadata.attn_norm,
        )?;

        Ok(Self {
            attn_norm,
            attn_wq: WeightData {
                weight: attn_wq,
                bias: attn_q_bias,
            },
            attn_wk: WeightData {
                weight: attn_wk,
                bias: attn_k_bias,
            },
            attn_wv: WeightData {
                weight: attn_wv,
                bias: attn_v_bias,
            },
            attn_wo: WeightData {
                weight: attn_wo,
                bias: attn_o_bias,
            },
            ffn_norm,
            ffn_hidden: WeightData {
                weight: ffn_hidden_weights,
                bias: ffn_hidden_bias,
            },
            ffn_output: WeightData {
                weight: ffn_output_weights,
                bias: ffn_output_bias,
            },
        })
    }

    fn find_prefix(path: &ModelPath, layer_idx: usize) -> super::Result<String> {
        for padding_len in 0..10 {
            let num = format!("{}{layer_idx}", "0".repeat(padding_len));
            // look for the attention qkv weights, arbitrarily; any of them should be okay
            let candidate = path.path(["transformer.", &num, ".qkv.weights.bin"]);
            if fs::exists(candidate)? {
                return Ok(num);
            }
        }
        Err(format!("couldn't find files for layer {layer_idx}").into())
    }
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
    let mut contents = BufReader::new(File::open(path)?);
    contents.read_exact(&mut contents_u8)?;
    let extra_bytes = contents.bytes().count();
    if extra_bytes > 0 {
        let s = format!(
            "got extra bytes in {path:?}. expected {expected_size} (for shape {size}), saw {expected_size} extra"
        );
        return Err(s.into());
    }

    Ok(contents_u8
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}
