use crate::bpe::Rank;
use crate::tensor::{Matrix, Tensor};
use crate::transformer::{Norm, TransformerBlock};
use std::fmt::{Display, Formatter};

pub struct ModelLoader {
    pub tok_embed: Matrix,
    pub pos_embed: Matrix,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: Norm,
}

pub struct Model {
    fwd: ModelLoader,
    tok_unembed: Matrix,
}

impl ModelLoader {
    pub fn initialize(self) -> Model {
        let tok_unembed = self.tok_embed.clone().t().contiguous();
        Model { fwd: self, tok_unembed }
    }
}

impl Model {
    pub fn apply(&self, seq: &Vec<Rank>) -> Result<Matrix, InferenceError> {
        let mut x = self.embed_inputs(seq)?;

        for transformer in &self.fwd.layers {
            x = transformer.apply(x);
        }

        x = self.fwd.final_norm.apply(&x);

        let unembedding = x.matmul(&self.tok_unembed);

        Ok(unembedding)
    }

    pub fn vocab_size(&self) -> usize {
        self.fwd.tok_embed.num_rows()
    }

    pub fn token_embedding_dim(&self) -> usize {
        self.fwd.tok_embed.num_cols()
    }

    pub fn max_seq_len(&self) -> usize {
        self.fwd.pos_embed.num_rows()
    }

    fn embed_inputs(&self, seq: &Vec<Rank>) -> Result<Matrix, InferenceError> {
        if seq.len() > self.max_seq_len() {
            return Err(InferenceError::MaxSeq);
        }
        let mut tok_embeddings = Tensor::new_matrix(seq.len(), self.token_embedding_dim());
        let mut pos_embeddings = Tensor::new_matrix(seq.len(), self.token_embedding_dim());
        for (seq_idx, &seq_tok) in seq.into_iter().enumerate() {
            // copy the right token embedding to tok_embeddings
            self.fwd.tok_embed.with_row([seq_tok.rank(), 0], |tok_embed| {
                tok_embeddings.set_row([seq_idx, 0], tok_embed);
            });
            // and the right pos embedding to pos_embeddings
            self.fwd.pos_embed.with_row([seq_idx, 0], |pos_embed| {
                pos_embeddings.set_row([seq_idx, 0], pos_embed);
            });
        }
        let input = tok_embeddings.add_tensor(&pos_embeddings);
        Ok(input)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InferenceError {
    MaxSeq,
}

impl Display for InferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::MaxSeq => write!(f, "max sequence length reached"),
        }
    }
}

impl std::error::Error for InferenceError {}
