use crate::cputensor::Matrix;
use crate::transformer::attention::Attention;
use crate::transformer::ffn::Ffn;
use crate::transformer::norm::Norm;

pub struct TransformerBlock {
    attention_norm: Norm,
    attention: Attention,
    ffn_norm: Norm,
    ffn: Ffn,
}

impl TransformerBlock {
    pub fn new(attention_norm: Norm, attention: Attention, ffn_norm: Norm, ffn: Ffn) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    pub fn mut_attn(&mut self) -> &'_ mut Attention {
        &mut self.attention
    }

    pub fn mut_attn_norm(&mut self) -> &'_ mut Norm {
        &mut self.attention_norm
    }

    pub fn mut_ffn(&mut self) -> &'_ mut Ffn {
        &mut self.ffn
    }

    pub fn mut_ffn_norm(&mut self) -> &'_ mut Norm {
        &mut self.ffn_norm
    }

    pub fn apply(&self, input: Matrix) -> Matrix {
        let pre_attn_norm = self.attention_norm.apply(&input);
        let attn = self.attention.apply(&pre_attn_norm);
        let attn_and_residual = attn.add_tensor(&input);

        let pre_ffn_norm = self.ffn_norm.apply(&attn_and_residual);
        let ffn = self.ffn.apply_matrix(pre_ffn_norm);
        ffn.add_tensor(&attn_and_residual)
    }
}
