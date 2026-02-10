use serde::{Deserialize, Serialize};

pub type Offsets = (usize, usize);

#[derive(Serialize, Deserialize, Default, Clone, Debug, Eq, PartialEq)]
pub struct TensorFileOffsets {
    pub bpe_merges: Offsets,
    pub bpe_encoder: Offsets,
    pub tok_embed: Offsets,
    pub pos_embed: Offsets,
    pub transformers: Vec<TransformerOffsets>,
    pub final_norm: NormOffsets,
}

#[derive(Serialize, Deserialize, Default, Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformerOffsets {
    pub before_attn_norm: NormOffsets,
    pub attn_qkv: MatrixOffsets,
    pub attn_wo: MatrixOffsets,

    pub before_ffn_norm: NormOffsets,
    pub ffn_hidden: MatrixOffsets,
    pub ffn_output: MatrixOffsets,
}

#[derive(Serialize, Deserialize, Default, Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormOffsets {
    pub scale: Offsets,
    pub bias: Offsets,
}

#[derive(Serialize, Deserialize, Default, Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixOffsets {
    pub weight: Offsets,
    pub bias: Offsets,
}
