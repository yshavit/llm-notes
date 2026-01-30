use crate::tensor::{Matrix, Tensor, matmul, matmul_batched_3, softmax};

pub struct Attention {
    embedding_dim: usize,
    num_heads: usize,
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
}

impl Attention {
    //noinspection RsAssertEqual -- I don't want assert_eq!, because I don't need to print the actual/expected
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        assert!(
            (embedding_dim / num_heads) * num_heads == embedding_dim,
            "embedding_dim ({embedding_dim}) must be a multiple of num_heads ({num_heads})"
        );
        Self {
            embedding_dim,
            num_heads,

            w_q: Tensor::new_matrix(embedding_dim, embedding_dim),
            w_k: Tensor::new_matrix(embedding_dim, embedding_dim),
            w_v: Tensor::new_matrix(embedding_dim, embedding_dim),
            w_o: Tensor::new_matrix(embedding_dim, embedding_dim),
        }
    }

    pub fn apply(&self, input: Matrix) -> Matrix {
        assert_eq!(
            input.num_cols(),
            self.embedding_dim,
            "expected input embedding {}, got {}",
            self.embedding_dim,
            input.shape(),
        );
        let (n, d, h) = (input.num_rows(), self.embedding_dim, self.num_heads);
        let qkv = |weight: &Matrix| -> Tensor<3> {
            let mut result = Tensor::new_matrix(n, d);
            matmul(&input, weight, &mut result);
            let mut result = result.reshape([n, h, d / h]);
            result = result.transposed(0, 1);
            result
        };

        // each are (h, n, d/h)
        let queries = qkv(&self.w_q);
        let keys = qkv(&self.w_k);
        let values = qkv(&self.w_v);

        let mut a = matmul_batched_3(&queries, &keys.transposed(1, 2));
        // divide by sqrt(d), and do softmax on the last dimension (d/h)
        a.multiply_scalar(1.0 / (d as f32).sqrt());
        for batch in 0..h {
            let mut batch_slice = a.matrix_slice_mut([batch, 0, 0]);
            for row in 0..batch_slice.num_rows() {
                batch_slice.mut_row(row, softmax)
            }
        }

        let mut attn = matmul_batched_3(&a, &values);
        // transpose from (h, n, d/h) to (n, h, d/h)
        attn = attn.transposed(1, 2);
        // reshape
        let attn = attn.reshape([n, d]);

        let mut output = Tensor::new_matrix(n, d);
        matmul(&attn, &self.w_o, &mut output);

        output
    }
}
