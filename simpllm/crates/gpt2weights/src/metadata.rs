use crate::path::{ModelPath, read_nicely};
use crate::{SIMPLLM_METADATA_FILE, TensorFileOffsets};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ModelShape {
    pub transformer: Vec<TransformerShape>,
    pub h_params: HParams,
    pub tensor_offsets: TensorFileOffsets,
    pub file_names: FileNames,
}

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct FileNames {
    pub tensors: String,
    pub bpe_merges: String,
    pub bpe_encodings: String,
}

#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct TransformerShape {
    pub ffn_hidden_layer_embed: usize,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct HParams {
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub eos_token_id: usize,
}

pub fn load_metadata(model: &ModelPath) -> Result<ModelShape, Box<dyn Error>> {
    let shape_reader = read_nicely(model, SIMPLLM_METADATA_FILE)?;
    let shape: ModelShape = serde_json::from_reader(shape_reader)?;

    Ok(shape)
}
