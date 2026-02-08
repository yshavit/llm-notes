use crate::path::{ModelPath, read_nicely};
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct ModelShape {
    pub layer: Vec<TransformerShape>,
    pub final_norm: NormShape,
    pub tok_embed: Vec<usize>,
    pub pos_embed: Vec<usize>,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct NormShape {
    pub bias: usize,
    pub scale: usize,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct TransformerShape {
    pub attn: AttentionShape,
    pub attn_norm: NormShape,
    pub ffn: FfnShape,
    pub ffn_norm: NormShape,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct AttentionShape {
    pub qkv: WeightsShape,
    pub output: WeightsShape,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct FfnShape {
    pub hidden: WeightsShape,
    pub output: WeightsShape,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct WeightsShape {
    pub weights: (usize, usize),
    pub bias: usize,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct HParams {
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
}

pub fn load_metadata(model: &ModelPath) -> Result<(ModelShape, HParams), Box<dyn Error>> {
    let shape_reader = read_nicely(&model.path("metadata.json"))?;
    let shape: ModelShape = serde_json::from_reader(shape_reader)?;

    let h_params_reader = read_nicely(&model.path("hparams.json"))?;
    let h_params: HParams = serde_json::from_reader(h_params_reader)?;

    Ok((shape, h_params))
}
