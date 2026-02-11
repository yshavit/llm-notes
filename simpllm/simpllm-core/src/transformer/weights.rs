use crate::tensor::{Matrix, Tensor, Tensor2D, TensorBackend, Vector};

pub struct MatrixAndBias<B: TensorBackend> {
    weights: Matrix<B>,
    bias: Vector<B>,
}

impl<B: TensorBackend> MatrixAndBias<B> {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weights: B::new_matrix(in_dim, out_dim),
            bias: B::new_vector(out_dim),
        }
    }

    pub fn in_dims(&self) -> usize {
        self.weights.num_rows()
    }

    pub fn out_dims(&self) -> usize {
        self.weights.num_cols()
    }

    pub fn weights(&self) -> &Matrix<B> {
        &self.weights
    }

    pub fn bias(&self) -> &Vector<B> {
        &self.bias
    }

    pub fn set(&mut self, weights: &Matrix<B>, bias: &Vector<B>) {
        self.weights.reset_values(&weights.flat_f32());
        self.bias.reset_values(&bias.flat_f32());
    }
}
