use crate::tensor::{Matrix, Tensor, Vector};

pub struct MatrixAndBias {
    weights: Matrix,
    bias: Vector,
}

impl MatrixAndBias {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weights: Tensor::new_matrix(in_dim, out_dim),
            bias: Tensor::new_vector(out_dim),
        }
    }

    pub fn in_dims(&self) -> usize {
        self.weights.num_rows()
    }

    pub fn out_dims(&self) -> usize {
        self.weights.num_cols()
    }

    pub fn weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn bias(&self) -> &Vector {
        &self.bias
    }

    pub fn set(&mut self, weights: &Matrix, bias: &Vector) {
        self.weights.reset_values(&weights.flat_f32());
        self.bias.reset_values(&bias.flat_f32());
    }
}
