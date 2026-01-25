use crate::ffn::activation::gelu;
use crate::tensor::{Matrix, Tensor, Vector, matmul};

pub struct LayerTransform {
    weights: Matrix,
    biases: Vec<f32>,
}

impl LayerTransform {
    pub fn new(in_dims: usize, out_dims: usize) -> Self {
        Self {
            weights: Tensor::new_matrix(in_dims, out_dims),
            biases: vec![0.; out_dims],
        }
    }

    pub fn apply(&mut self, inputs: &Vector, activations: &mut Vector) {
        // matmul will overwrite the activations, so we don't need to zero them out first
        matmul(inputs.as_row_matrix(), &self.weights, activations.as_row_matrix_mut());
        activations.mut_row([0], |row| {
            for i in 0..row.len() {
                row[i] = gelu(row[i] + self.biases[i])
            }
        })
    }
}
