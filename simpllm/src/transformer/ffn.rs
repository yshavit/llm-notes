use crate::tensor::{Matrix, Tensor, Vector, matmul};
use crate::transformer::activation::gelu;
use crate::transformer::weights::MatrixAndBias;

pub struct Ffn {
    layers_transforms: Vec<LayerTransform>,
}

impl Ffn {
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

    pub fn layer_mut(&mut self, layer: usize) -> &'_ mut MatrixAndBias {
        &mut self.layers_transforms[layer].mab
    }

    pub fn apply_matrix(&self, mut input: Matrix) -> Matrix {
        assert_eq!(input.num_cols(), self.in_dims(), "input dimensions");
        assert_eq!(input.num_cols(), self.out_dims(), "output dimensions");

        for row_idx in 0..input.num_rows() {
            let mut input_row = Tensor::new_vector(input.num_cols());
            input.with_row([row_idx, 0], |row| input_row.set_row([0], row));
            let ffn_result = self.apply(input_row);
            ffn_result.with_row([0], |in_row| {
                input.mut_row([row_idx, 0], |out| out.copy_from_slice(in_row));
            });
        }
        input
    }

    pub fn apply(&self, mut input: Vector) -> Vector {
        assert_eq!(
            input.len(),
            self.layers_transforms[0].mab.in_dims(),
            "expected input with dimension {}, got {}",
            input.len(),
            self.layers_transforms[0].mab.in_dims()
        );
        let mut transforms = self.layers_transforms.iter().peekable();
        while let Some(transform) = transforms.next() {
            let mut output = Tensor::new_vector(transform.mab.out_dims());
            transform.apply(&input, &mut output);
            if transforms.peek().is_some() {
                input.mut_row([0], |cols| cols.iter_mut().for_each(|v| *v = gelu(*v)))
            }
            input = output;
        }
        input
    }
}

struct LayerTransform {
    mab: MatrixAndBias,
}

impl LayerTransform {
    fn new(in_dims: usize, out_dims: usize) -> Self {
        Self {
            mab: MatrixAndBias::new(in_dims, out_dims),
        }
    }

    fn apply(&self, inputs: &Vector, activations: &mut Vector) {
        // matmul will overwrite the activations, so we don't need to zero them out first
        matmul(
            inputs.as_row_matrix(),
            self.mab.weights(),
            activations.as_row_matrix_mut(),
        );
        activations.mut_row([0], |row| {
            for i in 0..row.len() {
                row[i] = row[i] + self.mab.bias().get([i]);
            }
        })
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
        let mut ffn = Ffn::new(5, &[7], 6);
        for transform in &mut ffn.layers_transforms {
            transform.mab.set(
                &count_up([transform.mab.in_dims(), transform.mab.out_dims()]),
                &count_up([transform.mab.out_dims()]),
            );
        }

        let input = count_up([5]);

        let actual = ffn.apply(input);

        let mut expect = Tensor::new_vector(6);
        expect.reset_values(&[48441.0, 50850.0, 53259.0, 55668.0, 58077.0, 60486.0]);

        assert_f32_slice!(actual.as_f32(), expect.as_f32());
    }

    fn count_up<const R: usize>(shape: [usize; R]) -> Tensor<R> {
        let shape = Shape::from(shape);
        let vals: Vec<_> = (0..shape.num_elements()).map(|i| (i + 1) as f32).collect();
        let mut t = Tensor::new(shape);
        t.reset_values(&vals);
        t
    }
}
