use crate::ffn::activation::gelu;
use crate::tensor::{Matrix, Tensor, Vector, matmul};

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

    pub fn apply(&mut self, mut input: Vector) -> Vector {
        assert_eq!(
            input.len(),
            self.layers_transforms[0].in_dims(),
            "expected input with dimension {}, got {}",
            input.len(),
            self.layers_transforms[0].in_dims()
        );
        for transform in &self.layers_transforms {
            let mut output = Tensor::new_vector(transform.out_dims());
            transform.apply(&input, &mut output);
            input = output;
        }
        input
    }
}

struct LayerTransform {
    weights: Matrix,
    biases: Vec<f32>,
}

impl LayerTransform {
    fn new(in_dims: usize, out_dims: usize) -> Self {
        Self {
            weights: Tensor::new_matrix(in_dims, out_dims),
            biases: vec![0.; out_dims],
        }
    }

    fn in_dims(&self) -> usize {
        self.weights.num_rows()
    }

    fn out_dims(&self) -> usize {
        self.weights.num_cols()
    }

    fn apply(&self, inputs: &Vector, activations: &mut Vector) {
        // matmul will overwrite the activations, so we don't need to zero them out first
        matmul(inputs.as_row_matrix(), &self.weights, activations.as_row_matrix_mut());
        activations.mut_row([0], |row| {
            for i in 0..row.len() {
                row[i] = gelu(row[i] + self.biases[i])
            }
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::assert_f32_slice;

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
            let weight_size = transform.in_dims() * transform.out_dims();
            let bias_size = transform.out_dims();
            transform.set_weights(&count_up(weight_size), &count_up(bias_size));
        }

        let mut input = Tensor::new_vector(5);
        input.set_all(&count_up(5));

        let actual = ffn.apply(input);

        let mut expect = Tensor::new_vector(6);
        expect.set_all(&[48441.0, 50850.0, 53259.0, 55668.0, 58077.0, 60486.0]);

        assert_f32_slice!(actual.as_f32(), expect.as_f32());
    }

    fn count_up(num_elems: usize) -> Vec<f32> {
        (0..num_elems).map(|i| (i + 1) as f32).collect()
    }

    impl LayerTransform {
        pub fn set_weights(&mut self, values: &[f32], biases: &[f32]) {
            assert_eq!(
                values.len(),
                self.weights.shape().num_elements(),
                "can't set {} values to {} weights",
                values.len(),
                self.weights.shape()
            );
            assert_eq!(
                biases.len(),
                self.biases.len(),
                "can't set {} biases to {} weights",
                biases.len(),
                self.biases.len()
            );
            for (row_idx, row_values) in values.chunks(self.weights.num_cols()).enumerate() {
                self.weights.set_row([row_idx, 0], row_values);
            }
            self.biases.copy_from_slice(biases);
        }
    }
}
