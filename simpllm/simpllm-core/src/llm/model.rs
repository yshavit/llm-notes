use crate::bpe::Rank;
use crate::cputensor::LogitSampler;
use crate::tensor::{LayerNorm, Matrix, Tensor, Tensor2D, TensorBackend, TensorSlice};
use crate::transformer::{AttentionCache, TransformerBlock};
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

pub struct InferenceContext<B: TensorBackend> {
    attention_caches: Vec<AttentionCache<B>>,
    token_count: usize,
}

impl<B: TensorBackend> ModelLoader<B> {
    pub fn initialize(self) -> Model<B> {
        let tok_embed = B::Tensor::from_row_major(self.tok_embed.shape(), &self.tok_embed.flat_f32());
        let tok_unembed = tok_embed.transposed(0, 1).contiguous();
        Model { fwd: self, tok_unembed }
    }
}

impl<B: TensorBackend> Model<B> {
    pub fn new_inference_context(&self) -> InferenceContext<B> {
        InferenceContext {
            attention_caches: (0..self.fwd.layers.len()).map(|_| AttentionCache::default()).collect(),
            token_count: 0,
        }
    }

    pub fn apply(
        &self,
        seq: &[Rank],
        ctx: &mut InferenceContext<B>,
        logit_sampler: &Option<LogitSampler>,
    ) -> Result<Rank, InferenceError> {
        let mut x = self.embed_inputs(seq, ctx.token_count)?;

        for (i, transformer) in self.fwd.layers.iter().enumerate() {
            x = transformer.apply(x, &mut ctx.attention_caches[i]);
        }

        x = self.fwd.final_norm.apply(&x);

        let unembedding = x.matmul(&self.tok_unembed);

        let inferred_rank = match logit_sampler {
            Some(logit_sampler) => logit_sampler.get::<B>(unembedding),
            None => {
                // no sampling; just take argmax
                let last_logit = unembedding.num_rows() - 1;
                unembedding.slice_row([last_logit, 0], |row| {
                    // argmax
                    row.flat_f32()
                        .iter()
                        .enumerate()
                        .reduce(|acc, e| if e.1 > acc.1 { e } else { acc })
                        .map(|r| r.0)
                        .unwrap_or(0)
                })
            }
        };
        ctx.token_count += seq.len();

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

    fn embed_inputs(&self, seq: &[Rank], pos_offset: usize) -> Result<Matrix<B>, InferenceError> {
        if seq.len() + pos_offset > self.max_seq_len() {
            return Err(InferenceError::MaxSeq);
        }
        let mut tok_embeddings = B::new_matrix(seq.len(), self.token_embedding_dim());
        let mut pos_embeddings = B::new_matrix(seq.len(), self.token_embedding_dim());
        for (seq_idx, &seq_tok) in seq.iter().enumerate() {
            // copy the right token embedding to tok_embeddings
            self.fwd.tok_embed.slice_row([seq_tok.rank(), 0], |tok_embed| {
                tok_embeddings.set_slice([seq_idx, 0], tok_embed);
            });
            // and the right pos embedding to pos_embeddings
            // TODO add the gotcha of pos_offset to the book
            self.fwd.pos_embed.slice_row([seq_idx + pos_offset, 0], |pos_embed| {
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
