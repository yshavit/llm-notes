use crate::load::path::ModelPath;
use serde::Deserialize;

pub type Num = usize;

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct ModelMetadata {
    pub layer: Vec<Transformer>,
    pub final_norm: Norm,
    pub tok_embed: Vec<Num>,
    pub pos_embed: Vec<Num>,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct Norm {
    pub bias: Num,
    pub scale: Num,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct Transformer {
    pub attn: Attention,
    pub attn_norm: Norm,
    pub ffn: Ffn,
    pub ffn_norm: Norm,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct Attention {
    pub qkv: Weights,
    pub output: Weights,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct Ffn {
    pub hidden: Weights,
    pub output: Weights,
}

#[derive(Deserialize, Debug, Eq, PartialEq)]
pub struct Weights {
    pub weights: (usize, usize),
    pub bias: Num,
}

pub fn load_metadata(model: &ModelPath) -> super::Result<ModelMetadata> {
    let reader = model.read("metadata.json")?;
    Ok(serde_json::from_reader(reader)?)
}
