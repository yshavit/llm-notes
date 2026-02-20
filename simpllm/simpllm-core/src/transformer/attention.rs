use crate::tensor::{Matrix, Tensor, Tensor2D, TensorBackend};
use crate::transformer::weights::MatrixAndBias;

pub struct Attention<B: TensorBackend> {
    num_heads: usize,

    w_qkv: MatrixAndBias<B>,
    w_o: MatrixAndBias<B>,
}

impl<B: TensorBackend> Attention<B> {
    //noinspection RsAssertEqual -- I don't want assert_eq!, because I don't need to print the actual/expected
    pub fn new(w_qkv: MatrixAndBias<B>, w_o: MatrixAndBias<B>, num_heads: usize) -> Self {
        let qkv_embedding_dim = w_qkv.in_dims();
        let o_embedding_dim = w_o.in_dims();
        assert_eq!(
            qkv_embedding_dim, o_embedding_dim,
            "QKV and output must have same embedding dimension"
        );
        assert_eq!(qkv_embedding_dim * 3, w_qkv.out_dims(), "QKV must be [d x 3d]");
        Self { num_heads, w_qkv, w_o }
    }

    fn embedding_dim(&self) -> usize {
        self.w_qkv.in_dims()
    }

    pub fn apply(&self, input: &Matrix<B>) -> Matrix<B> {
        assert_eq!(
            input.num_cols(),
            self.embedding_dim(),
            "expected input embedding {}, got {}",
            self.embedding_dim(),
            input.shape(),
        );
        let (n, d, h) = (input.num_rows(), self.embedding_dim(), self.num_heads);

        // TODO in practice, these are brought in as a single matrix that's the three of these concatenated.
        // each are (h, n, d/h)
        let [queries, keys, values] = {
            let combined = input.matmul(self.w_qkv.weights()).add(self.w_qkv.bias());

            // Combined is (N x 3d). Add the biases before reshaping.

            let split = combined.split::<3>(1);
            split.map(|m| {
                // Each split is [n x d]. Reshape it to separate the d dimension by head, and then transpose it so the
                // head (not seq) is the batch dimension.
                let reshaped = m.reshape([n, h, d / h]);
                reshaped.transposed(0, 1)
            })
        };

        let mut a = queries.matmul(&keys.transposed(1, 2));
        // divide by sqrt(d/h), and do softmax on the last dimension (d/h)
        let dim_per_head = d / h;
        a.multiply_scalar(1.0 / (dim_per_head as f32).sqrt());

        // causal attention, then softmax
        let causal_mask = B::lower_triangle(n);
        a = a.add(&causal_mask);
        a = a.softmax();

        let mut attn = a.matmul(&values);
        // transpose from (h, n, d/h) to (n, h, d/h)
        attn = attn.transposed(0, 1);
        // reshape
        let attn = attn.contiguous().reshape([n, d]);

        // apply the weights
        attn.matmul(self.w_o.weights()).add(self.w_o.bias())
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_f32_slice;
    use crate::cputensor::CpuTensor;
    use crate::tensor::{Tensor, Tensor2D};
    use crate::transformer::attention::Attention;
    use crate::transformer::weights::MatrixAndBias;

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
    /// causal_mask = torch.triu(torch.ones(n_tokens, n_tokens), diagonal=1).bool()
    /// output, attn_weights = model(input_tensor, input_tensor, input_tensor, attn_mask=causal_mask, need_weights=True)
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
    ///
    /// [[0.011457345448434353,
    ///   0.011363562196493149,
    ///   0.011271310970187187,
    ///   0.011180554516613483,
    ///   0.011091256514191628,
    ///   0.011003383435308933],
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
        use crate::cputensor::CpuBackend;

        let embedding_dim: usize = 6;
        let n_heads = 3;
        let n_tokens = 2;

        let weights_size = embedding_dim.pow(2);
        let mut counter_data = (1..).map(|i| 1.0 / (i as f32));
        let mut counter = |n: usize| -> Vec<f32> { counter_data.by_ref().take(n).collect::<Vec<_>>() };

        use crate::tensor::TensorBackend;

        let qkv_mab = {
            // PyTorch stores QKV as [3*d_model, d_model], we store as [d_model, 3*d_model]
            // So we need to create the PyTorch layout and transpose it
            let qkv_weights = CpuTensor::from_row_major([embedding_dim * 3, embedding_dim], &counter(weights_size * 3))
                .transposed(0, 1)
                .contiguous();
            let zero_bias_qkv = CpuBackend::new_vector(embedding_dim * 3);
            MatrixAndBias::new(qkv_weights, zero_bias_qkv)
        };

        let o_mab = {
            let o_weights = CpuTensor::from_row_major([embedding_dim, embedding_dim], &counter(weights_size));
            let zero_bias_o = CpuBackend::new_vector(embedding_dim);
            MatrixAndBias::new(o_weights, zero_bias_o)
        };

        let tokens = CpuTensor::from_row_major(
            [embedding_dim * n_tokens],
            &(0..(n_tokens * embedding_dim))
                .map(|i| (i + 1) as f32)
                .collect::<Vec<_>>(),
        );

        let attention: Attention<CpuBackend> = Attention::new(qkv_mab, o_mab, n_heads);

        let tokens = tokens.reshape([n_tokens, embedding_dim]);

        let output = attention.apply(&tokens);

        let expected = [
            [
                0.011_457_345,
                0.011_363_562,
                0.011_271_311,
                0.011_180_554_5,
                0.011_091_256_5,
                0.011_003_383,
            ],
            [
                0.029_986_324,
                0.029_739_94,
                0.029_497_596,
                0.029_259_196,
                0.029_024_64,
                0.028_793_84,
            ],
        ];

        assert_eq!(output.num_rows(), expected.len());
        let actual_as_vec: Vec<Vec<f32>> = output.flat_f32().chunks_exact(embedding_dim).map(Vec::from).collect();
        for row in 0..expected.len() {
            assert_f32_slice!(&actual_as_vec[row], &expected[row]);
        }
    }
}
