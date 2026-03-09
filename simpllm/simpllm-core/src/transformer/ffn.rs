use crate::tensor::{Matrix, Tensor, Tensor2D, TensorBackend};
use crate::transformer::weights::MatrixAndBias;

pub struct Ffn<B: TensorBackend> {
    layers_transforms: Vec<MatrixAndBias<B>>,
}

impl<B: TensorBackend> Ffn<B> {
    pub fn new(layers_transforms: Vec<MatrixAndBias<B>>) -> Self {
        let mut layers_iter = layers_transforms.iter().enumerate();
        let mut out_dim = {
            let (_, first_layer) = layers_iter.next().expect("layer_transforms may not be empty");
            first_layer.out_dims()
        };
        for (layer_num, layer) in layers_iter {
            assert_eq!(layer.in_dims(), out_dim, "at layer {layer_num}");
            out_dim = layer.out_dims();
        }
        Self { layers_transforms }
    }

    fn in_dims(&self) -> usize {
        self.layers_transforms[0].in_dims()
    }

    pub fn layer_mut(&mut self, layer: usize) -> &'_ mut MatrixAndBias<B> {
        &mut self.layers_transforms[layer]
    }

    pub fn apply(&self, mut input: Matrix<B>) -> Matrix<B> {
        assert_eq!(input.num_cols(), self.in_dims(), "input dimensions");
        // MYSTMD::FFN START
        let mut transforms = self.layers_transforms.iter().peekable();
        while let Some(transform) = transforms.next() {
            // Apply the transformation's weights and bias
            let mut output = input.matmul(transform.weights()).add(transform.bias());

            // If this isn't the last transformation, also apply the activation function.
            if transforms.peek().is_some() {
                output = output.gelu();
            }
            input = output;
        }
        // MYSTMD::FFN END
        input
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::assert_f32_slice;
    use crate::cputensor::CpuTensor;
    use crate::tensor::Shape;

    /// Compares against a reference pytorch implementation.
    ///
    /// The pytorch FFN uses:
    /// - input dimension 5
    /// - one hidden layer, dimension 7
    /// - output dimension 6
    /// - tanh-approximated activation
    ///
    /// The code is:
    ///
    /// ```python
    /// import torch
    /// import torch.nn as nn
    ///
    /// class FFN(nn.Module):
    ///     def __init__(self):
    ///         super(FFN, self).__init__()
    ///         self.fc1 = nn.Linear(5, 7)
    ///         self.gelu = nn.GELU(approximate='tanh')
    ///         self.fc2 = nn.Linear(7, 6)
    ///
    ///     def forward(self, x):
    ///         x = self.fc1(x)
    ///         x = self.gelu(x)
    ///         x = self.fc2(x)
    ///         return x
    ///
    /// model = FFN()
    ///
    /// # Set fc1 weights and biases. Each set of parameters will have values 1-N
    /// with torch.no_grad():
    ///     # Pytorch defines layer parameters as transposed from the matrix multiplication.
    ///     # So, I'll create them as they'd look in the matrix multiplication, and then use the transposition.
    ///     model.fc1.weight = nn.Parameter(torch.arange(1, 36).float().reshape(5, 7).T)
    ///     model.fc1.bias = nn.Parameter(torch.arange(1, 8).float())
    ///
    ///     model.fc2.weight = nn.Parameter(torch.arange(1, 43).float().reshape(7, 6).T)
    ///     model.fc2.bias = nn.Parameter(torch.arange(1, 7).float())
    ///
    /// # Fixed input, also 1-N
    /// input_tensor = torch.arange(1, 6).float().unsqueeze(0)  # Shape: [1, 5]
    ///
    /// output = model(input_tensor)
    /// print("Output:", output.tolist())
    /// ```
    ///
    /// Output is:
    ///
    /// ```text
    /// Output: [[48441.0, 50850.0, 53259.0, 55668.0, 58077.0, 60486.0]]
    /// ```
    ///
    #[test]
    fn compare_against_pytorch() {
        use crate::cputensor::CpuBackend;

        let layers = vec![
            MatrixAndBias::new(count_up([5, 7]), count_up([7])),
            MatrixAndBias::new(count_up([7, 6]), count_up([6])),
        ];

        let ffn: Ffn<CpuBackend> = Ffn::new(layers);

        let input = count_up([5]).reshape([1, 5]);

        let actual = ffn.apply(input);

        let expect = CpuTensor::from_row_major([6], &[48441.0, 50850.0, 53259.0, 55668.0, 58077.0, 60486.0]);

        assert_f32_slice!(actual.flat_f32().as_ref(), expect.flat_f32().as_ref());
    }

    fn count_up<const R: usize>(shape: [usize; R]) -> CpuTensor<R> {
        let shape = Shape::from(shape);
        let vals: Vec<_> = (0..shape.num_elements()).map(|i| (i + 1) as f32).collect();
        CpuTensor::from_row_major(shape, &vals)
    }
}
