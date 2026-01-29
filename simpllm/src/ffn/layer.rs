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

    pub fn in_dims(&self) -> usize {
        self.weights.num_rows()
    }

    pub fn out_dims(&self) -> usize {
        self.biases.len()
    }

    pub fn apply(&self, inputs: &Vector, activations: &mut Vector) {
        // matmul will overwrite the activations, so we don't need to zero them out first
        matmul(inputs.as_row_matrix(), &self.weights, activations.as_row_matrix_mut());
        activations.mut_row([0], |row| {
            for i in 0..row.len() {
                row[i] = gelu(row[i] + self.biases[i])
            }
        })
    }

    #[cfg(test)]
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
