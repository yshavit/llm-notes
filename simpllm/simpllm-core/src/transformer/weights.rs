use crate::tensor::{Matrix, Tensor2D, TensorBackend, Vector};

pub struct MatrixAndBias<B: TensorBackend> {
    weights: Matrix<B>,
    bias: Vector<B>,
}

impl<B: TensorBackend> MatrixAndBias<B> {
    pub fn new(weights: Matrix<B>, bias: Vector<B>) -> Self {
        Self { weights, bias }
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
}
