use crate::tensor::{Matrix, Tensor, Vector, matmul, matmul_batched_3, softmax};

pub struct Attention {
    embedding_dim: usize,
    num_heads: usize,
    w_q: MatrixAndBias,
    w_k: MatrixAndBias,
    w_v: MatrixAndBias,
    w_o: MatrixAndBias,
}

struct MatrixAndBias {
    weights: Matrix,
    // TODO need to update the book to include this!
    bias: Vector,
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

            w_q: MatrixAndBias {
                weights: Tensor::new_matrix(embedding_dim, embedding_dim),
                bias: Tensor::new_vector(embedding_dim),
            },
            w_k: MatrixAndBias {
                weights: Tensor::new_matrix(embedding_dim, embedding_dim),
                bias: Tensor::new_vector(embedding_dim),
            },
            w_v: MatrixAndBias {
                weights: Tensor::new_matrix(embedding_dim, embedding_dim),
                bias: Tensor::new_vector(embedding_dim),
            },
            w_o: MatrixAndBias {
                weights: Tensor::new_matrix(embedding_dim, embedding_dim),
                bias: Tensor::new_vector(embedding_dim),
            },
        }
    }

    pub fn apply(&self, input: &Matrix) -> Matrix {
        assert_eq!(
            input.num_cols(),
            self.embedding_dim,
            "expected input embedding {}, got {}",
            self.embedding_dim,
            input.shape(),
        );
        let (n, d, h) = (input.num_rows(), self.embedding_dim, self.num_heads);
        let qkv = |weight: &MatrixAndBias| -> Tensor<3> {
            let mut result = Tensor::new_matrix(n, d);
            matmul(input, &weight.weights, &mut result);
            // result is (N x d). Add the biases before reshaping.
            result.add_broadcasted_vector(&weight.bias);

            let result = result.reshape([n, h, d / h]);
            result.transposed(0, 1)
        };

        // each are (h, n, d/h)
        let queries = qkv(&self.w_q);
        let keys = qkv(&self.w_k);
        let values = qkv(&self.w_v);

        let mut a = matmul_batched_3(&queries, &keys.transposed(1, 2));
        // divide by sqrt(d/h), and do softmax on the last dimension (d/h)
        let dim_per_head = d / h;
        a.multiply_scalar(1.0 / (dim_per_head as f32).sqrt());
        for batch in 0..h {
            let mut batch_slice = a.matrix_slice_mut([batch, 0, 0]);
            for row in 0..batch_slice.num_rows() {
                batch_slice.mut_row(row, softmax)
            }
        }

        let mut attn = matmul_batched_3(&a, &values);
        // transpose from (h, n, d/h) to (n, h, d/h)
        attn = attn.transposed(0, 1);
        // reshape
        let attn = attn.contiguous().reshape([n, d]);

        let mut output = Tensor::new_matrix(n, d);
        matmul(&attn, &self.w_o.weights, &mut output);
        output.add_broadcasted_vector(&self.w_o.bias);

        output
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_f32_slice;
    use crate::tensor::Tensor;
    use crate::transformer::attention::Attention;

    /// Compares against a reference pytorch implementation.
    ///
    /// ```python
    /// import torch
    /// import torch.nn as nn
    /// from pprint import pprint
    ///
    /// # Parameters
    /// d_model = 6
    /// n_heads = 3
    /// n_tokens = 2
    ///
    /// # Create model
    /// model = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, bias=False, batch_first=True)
    ///
    /// # Generator for 1/1, 1/2, 1/3, ...
    /// def value_generator():
    ///     i = 1
    ///     while True:
    ///         yield 1.0 / i
    ///         i += 1
    ///
    /// gen = value_generator()
    ///
    /// # Set weights using the generator
    /// with torch.no_grad():
    ///     num_elements = d_model * d_model
    ///
    ///     # in_proj_weight: [3*d_model, d_model]
    ///     total_qkv_elements = 3 * num_elements
    ///     model.in_proj_weight = nn.Parameter(
    ///         torch.tensor([next(gen) for _ in range(total_qkv_elements)]).reshape(3 * d_model, d_model)
    ///     )
    ///
    ///     # out_proj.weight
    ///     model.out_proj.weight = nn.Parameter(
    ///         torch.tensor([next(gen) for _ in range(num_elements)]).reshape(d_model, d_model).T
    ///     )
    ///
    /// # Input tokens: [1,2,3,4,5,6] for token 1, [7,8,9,10,11,12] for token 2
    /// # (Pytorch expects a batch dimension, so we'll give it a batch size of 1)
    /// input_tensor = torch.arange(1, n_tokens * d_model + 1).float().reshape(1, n_tokens, d_model)
    ///
    /// output, attn_weights = model(input_tensor, input_tensor, input_tensor)
    ///
    /// print("Input:  ", input_tensor.tolist())
    /// print("Output: ")
    /// pprint(output.squeeze(0).tolist())
    /// ```
    ///
    /// Output is:
    ///
    /// ```text
    /// Input:   [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]
    /// Output:
    /// [[0.027653157711029053,
    ///   0.027424853295087814,
    ///   0.02720031887292862,
    ///   0.02697945199906826,
    ///   0.026762165129184723,
    ///   0.026548368856310844],
    ///  [0.029986323788762093,
    ///   0.02973994053900242,
    ///   0.029497595503926277,
    ///   0.029259195551276207,
    ///   0.02902464009821415,
    ///   0.02879383973777294]]
    /// ```
    ///
    #[test]
    fn compare_against_pytorch() {
        let embedding_dim = 6;
        let n_heads = 3;
        let n_tokens = 2;

        let mut attention = Attention::new(embedding_dim, n_heads);
        let weights_size = embedding_dim.pow(2);
        let mut counter_data = (1..).map(|i| 1.0 / (i as f32));
        let mut counter = |n: usize| -> Vec<f32> { counter_data.by_ref().take(weights_size).collect::<Vec<_>>() };

        // pytorch transposes the q/k/v matrices internally, within its transforms. We don't, so I'll transpose them
        // here instead.
        attention.w_q.weights.reset_values(&counter(weights_size));
        attention.w_q.weights = attention.w_q.weights.t().contiguous();

        attention.w_k.weights.reset_values(&counter(weights_size));
        attention.w_k.weights = attention.w_k.weights.t().contiguous();

        attention.w_v.weights.reset_values(&counter(weights_size));
        attention.w_v.weights = attention.w_v.weights.t().contiguous();

        attention.w_o.weights.reset_values(&counter(weights_size));

        let mut tokens = Tensor::new([embedding_dim * n_tokens]);
        tokens.reset_values(
            &(0..(n_tokens * embedding_dim))
                .map(|i| (i + 1) as f32)
                .collect::<Vec<_>>(),
        );
        let tokens = tokens.reshape([n_tokens, embedding_dim]);

        let output = attention.apply(&tokens);

        let expected = vec![
            [
                0.027653157711029053,
                0.027424853295087814,
                0.02720031887292862,
                0.02697945199906826,
                0.026762165129184723,
                0.026548368856310844,
            ],
            [
                0.029986323788762093,
                0.02973994053900242,
                0.029497595503926277,
                0.029259195551276207,
                0.02902464009821415,
                0.02879383973777294,
            ],
        ];

        assert_eq!(output.num_rows(), expected.len());
        let actual_as_vec = output.to_f32();
        for row in 0..expected.len() {
            assert_f32_slice!(&actual_as_vec[row], &expected[row]);
        }
    }
}
