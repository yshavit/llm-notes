use crate::path::{read_nicely, ModelFile, ModelPath};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ModelShape {
    pub transformer: Vec<TransformerShape>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct TransformerShape {
    pub ffn_hidden_layer_embed: usize,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct WeightsShape {
    pub weights: (usize, usize),
    pub bias: usize,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct HParams {
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub eos_token_id: usize,
}

pub fn load_metadata(model: &ModelPath) -> Result<(ModelShape, HParams), Box<dyn Error>> {
    let shape_reader = read_nicely(model, ModelFile::MetadataJson)?;
    let shape: ModelShape = serde_json::from_reader(shape_reader)?;

    let h_params_reader = read_nicely(model, ModelFile::HParamsJson)?;
    let h_params: HParams = serde_json::from_reader(h_params_reader)?;

    Ok((shape, h_params))
}
