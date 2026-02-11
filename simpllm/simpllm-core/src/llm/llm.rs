use crate::bpe::Rank;
use crate::cputensor::LogitSampler;
use crate::tensor::{LayerNorm, Matrix, Tensor, Tensor2D, TensorBackend};
use crate::transformer::TransformerBlock;
use std::fmt::{Display, Formatter};

pub struct ModelLoader<B: TensorBackend> {
    pub tok_embed: Matrix<B>,
    pub pos_embed: Matrix<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub final_norm: B::LayerNorm,
    pub eos_token: usize,
}

pub struct Model<B: TensorBackend> {
    fwd: ModelLoader<B>,
    tok_unembed: Matrix<B>,
}

impl<B: TensorBackend> ModelLoader<B> {
    pub fn initialize(self) -> Model<B> {
        let mut tok_embed = B::Tensor::new(self.tok_embed.shape());
        tok_embed.reset_values(&self.tok_embed.flat_f32());
        let tok_unembed = tok_embed.transposed(0, 1).contiguous();
        Model { fwd: self, tok_unembed }
    }
}

impl<B: TensorBackend> Model<B> {
    pub fn apply(&self, seq: &Vec<Rank>, logit_sampler: &Option<LogitSampler>) -> Result<Rank, InferenceError> {
        let mut x = self.embed_inputs(seq)?;

        for transformer in &self.fwd.layers {
            x = transformer.apply(x);
        }

        x = self.fwd.final_norm.apply(&x);

        let unembedding = x.matmul(&self.tok_unembed);

        let inferred_rank = match logit_sampler {
            Some(logit_sampler) => logit_sampler.get::<B>(unembedding),
            None => {
                // no sampling; just take argmax
                unembedding.with_row([unembedding.num_rows() - 1, 0], |row| {
                    // argmax
                    row.iter()
                        .enumerate()
                        .reduce(|acc, e| if e.1 > acc.1 { e } else { acc })
                        .map(|r| r.0)
                        .unwrap_or(0)
                })
            }
        };

        Ok(inferred_rank.into())
    }

    pub fn eos(&self) -> usize {
        self.fwd.eos_token
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

    fn embed_inputs(&self, seq: &Vec<Rank>) -> Result<Matrix<B>, InferenceError> {
        if seq.len() > self.max_seq_len() {
            return Err(InferenceError::MaxSeq);
        }
        let mut tok_embeddings = B::new_matrix(seq.len(), self.token_embedding_dim());
        let mut pos_embeddings = B::new_matrix(seq.len(), self.token_embedding_dim());
        for (seq_idx, &seq_tok) in seq.into_iter().enumerate() {
            // copy the right token embedding to tok_embeddings
            self.fwd.tok_embed.slice_row([seq_tok.rank(), 0], |tok_embed| {
                tok_embeddings.set_slice([seq_idx, 0], tok_embed);
            });
            // and the right pos embedding to pos_embeddings
            self.fwd.pos_embed.slice_row([seq_idx, 0], |pos_embed| {
                pos_embeddings.set_slice([seq_idx, 0], pos_embed);
            });
        }
        let input = tok_embeddings.add(&pos_embeddings);
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
