use crate::tensor::Matrix;
use crate::transformer::{Norm, TransformerBlock};

pub struct Model {
    pub tok_embed: Matrix,
    pub pos_embed: Matrix,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: Norm,
}

impl Model {}
