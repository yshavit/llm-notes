use crate::tensor::{Matrix, Tensor, matmul};
use crate::transformer::{Norm, TransformerBlock};
use std::fmt::{Display, Formatter};
use tiktoken_rs::Rank;

pub struct Model {
    pub tok_embed: Matrix,
    pub pos_embed: Matrix,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: Norm,
}

impl Model {
    pub fn apply(&self, seq: &Vec<Rank>) -> Result<Matrix, InferenceError> {
        let mut x = self.embed_inputs(seq)?;

        for transformer in &self.layers {
            x = transformer.apply(x);
        }

        x = self.final_norm.apply(&x);

        let mut unembedding = Tensor::new_matrix(x.num_rows(), self.vocab_size());
        matmul(&x, &self.tok_embed.clone().t(), &mut unembedding);

        Ok(unembedding)
    }

    pub fn vocab_size(&self) -> usize {
        self.tok_embed.num_rows()
    }

    pub fn token_embedding_dim(&self) -> usize {
        self.tok_embed.num_cols()
    }

    pub fn max_seq_len(&self) -> usize {
        self.pos_embed.num_rows()
    }

    fn embed_inputs(&self, seq: &Vec<Rank>) -> Result<Matrix, InferenceError> {
        if seq.len() > self.max_seq_len() {
            return Err(InferenceError::MaxSeq);
        }
        let mut input = Tensor::new_matrix(seq.len(), self.token_embedding_dim());
        for (seq_idx, &seq_tok) in seq.into_iter().enumerate() {
            let seq_tok_usize: usize = seq_tok.try_into().map_err(|_| InferenceError::TokOutOfRange(seq_tok))?;
            self.tok_embed.with_row([seq_tok_usize, 0], |tok_embed| {
                input.set_row([seq_idx, 0], tok_embed);
            });
            self.pos_embed.with_row([seq_idx, 0], |pos_embed| {
                input.mut_row([seq_idx, 0], |input_embed| {
                    assert_eq!(pos_embed.len(), input_embed.len());
                    for i in 0..pos_embed.len() {
                        input_embed[i] += pos_embed[i];
                    }
                })
            })
        }
        Ok(input)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InferenceError {
    MaxSeq,
    TokOutOfRange(u32),
}

impl Display for InferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::MaxSeq => write!(f, "max sequence length reached"),
            InferenceError::TokOutOfRange(tok) => write!(f, "token is out of range: {tok}"),
        }
    }
}

impl std::error::Error for InferenceError {}
