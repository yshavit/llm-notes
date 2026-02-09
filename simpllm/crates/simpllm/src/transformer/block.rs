use crate::tensor::{Matrix, Tensor, TensorBackend};
use crate::transformer::attention::Attention;
use crate::transformer::ffn::Ffn;
use crate::transformer::norm::Norm;

pub struct TransformerBlock<B: TensorBackend> {
    attention_norm: Norm<B>,
    attention: Attention<B>,
    ffn_norm: Norm<B>,
    ffn: Ffn<B>,
}

impl<B: TensorBackend> TransformerBlock<B> {
    pub fn new(attention_norm: Norm<B>, attention: Attention<B>, ffn_norm: Norm<B>, ffn: Ffn<B>) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    pub fn mut_attn(&mut self) -> &'_ mut Attention<B> {
        &mut self.attention
    }

    pub fn mut_attn_norm(&mut self) -> &'_ mut Norm<B> {
        &mut self.attention_norm
    }

    pub fn mut_ffn(&mut self) -> &'_ mut Ffn<B> {
        &mut self.ffn
    }

    pub fn mut_ffn_norm(&mut self) -> &'_ mut Norm<B> {
        &mut self.ffn_norm
    }

    pub fn apply(&self, input: Matrix<B>) -> Matrix<B> {
        let pre_attn_norm = self.attention_norm.apply(&input);
        let attn = self.attention.apply(&pre_attn_norm);
        let attn_and_residual = attn.add(input);

        let pre_ffn_norm = self.ffn_norm.apply(&attn_and_residual);
        let ffn = self.ffn.apply_matrix(pre_ffn_norm);
        ffn.add(attn_and_residual)
    }
}
