use crate::tensor::{Matrix, Shape, Tensor, Tensor2D, TensorBackend};
use crate::transformer::weights::MatrixAndBias;

pub struct Ffn<B: TensorBackend> {
    layers_transforms: Vec<LayerTransform<B>>,
}

impl<B: TensorBackend> Ffn<B> {
    pub fn new(mut in_dim: usize, hidden_layer_dims: &[usize], out_dim: usize) -> Self {
        let mut layers = Vec::with_capacity(hidden_layer_dims.len() + 1);
        for &out_dim in hidden_layer_dims.iter().chain(std::iter::once(&out_dim)) {
            layers.push(LayerTransform::new(in_dim, out_dim));
            in_dim = out_dim;
        }
        Self {
            layers_transforms: layers,
        }
    }

    fn in_dims(&self) -> usize {
        self.layers_transforms[0].mab.in_dims()
    }

    fn out_dims(&self) -> usize {
        self.layers_transforms[self.layers_transforms.len() - 1].mab.out_dims()
    }

    pub fn layer_mut(&mut self, layer: usize) -> &'_ mut MatrixAndBias<B> {
        &mut self.layers_transforms[layer].mab
    }

    pub fn apply_matrix(&self, input: Matrix<B>) -> Matrix<B> {
        assert_eq!(input.num_cols(), self.in_dims(), "input dimensions");
        assert_eq!(input.num_cols(), self.out_dims(), "output dimensions");

        // TODO in the book, I talk about this being a 1xD matrix, and then applied separately to each row.
        //   But it can (and should!) be done as an RxD matrix, as done here
        self.apply(input)
    }

    pub fn apply(&self, mut input: Matrix<B>) -> Matrix<B> {
        let n_rows = input.num_rows();
        let mut transforms = self.layers_transforms.iter().peekable();
        while let Some(transform) = transforms.next() {
            assert_eq!(input.shape(), Shape::new([n_rows, transform.mab.in_dims()]));
            let mut output = transform.apply(&input);
            assert_eq!(output.shape(), Shape::new([n_rows, transform.mab.out_dims()]));
            if transforms.peek().is_some() {
                output = output.gelu();
            }
            input = output;
        }
        input
    }
}

struct LayerTransform<B: TensorBackend> {
    mab: MatrixAndBias<B>,
}

impl<B: TensorBackend> LayerTransform<B> {
    fn new(in_dims: usize, out_dims: usize) -> Self {
        Self {
            mab: MatrixAndBias::new(in_dims, out_dims),
        }
    }

    fn apply(&self, inputs: &Matrix<B>) -> Matrix<B> {
        let activations = inputs.matmul(self.mab.weights());
        activations.add(self.mab.bias())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::assert_f32_slice;
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

        let mut ffn: Ffn<CpuBackend> = Ffn::new(5, &[7], 6);
        for transform in &mut ffn.layers_transforms {
            transform.mab.set(
                &count_up([transform.mab.in_dims(), transform.mab.out_dims()]),
                &count_up([transform.mab.out_dims()]),
            );
        }

        let input = count_up([5]).reshape([1, 5]);

        let actual = ffn.apply(input);

        let mut expect = CpuBackend::new_vector(6);
        expect.reset_values(&[48441.0, 50850.0, 53259.0, 55668.0, 58077.0, 60486.0]);

        assert_f32_slice!(actual.flat_f32().as_ref(), expect.flat_f32().as_ref());
    }

    fn count_up<const R: usize>(shape: [usize; R]) -> <crate::cputensor::CpuBackend as TensorBackend>::Tensor<R> {
        let shape = Shape::from(shape);
        let vals: Vec<_> = (0..shape.num_elements()).map(|i| (i + 1) as f32).collect();
        let mut t = <crate::cputensor::CpuBackend as TensorBackend>::Tensor::new(shape);
        t.reset_values(&vals);
        t
    }
}
