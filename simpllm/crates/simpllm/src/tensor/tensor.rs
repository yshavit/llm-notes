use crate::tensor::{Shape, TensorBackend};
use std::borrow::Cow;

pub trait Tensor<const R: usize>: Sized + Clone + Send + Sync {
    type Backend: TensorBackend;

    fn new(shape: impl Into<Shape<R>>) -> Self;
    fn shape(&self) -> Shape<R>;

    fn reset_values(&mut self, values: &[f32]);

    fn reshape<const R2: usize>(self, new_shape: impl Into<Shape<R2>>) -> <Self::Backend as TensorBackend>::Tensor<R2>;
    fn split<const S: usize>(self, dim: usize) -> [<Self::Backend as TensorBackend>::Tensor<R>; S];
    fn transposed(self, dim0: usize, dim1: usize) -> Self;
    fn contiguous(self) -> Self;
    fn matmul(&self, other: &Self) -> Self;

    fn get(&self, indices: [usize; R]) -> f32;
    fn with_row<X>(&self, indices: [usize; R], f: impl FnOnce(&[f32]) -> X) -> X;

    fn mut_row<X>(&mut self, indices: [usize; R], f: impl FnOnce(&mut [f32]) -> X) -> X;
    fn flat_f32(&self) -> Cow<'_, [f32]>;
    fn gelu(self) -> Self;
    fn softmax(self) -> Self;

    fn multiply_scalar(&mut self, factor: f32);
    fn add<const R2: usize>(self, other: &<Self::Backend as TensorBackend>::Tensor<R2>) -> Self;
    fn set_row(&mut self, indices: [usize; R], values: &[f32]);
}

pub trait Tensor2D: Tensor<2> {
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
    fn mut_rows(&mut self, f: impl Fn(usize, &mut [f32]) + Sync);
}

impl<T: Tensor<2>> Tensor2D for T {
    fn num_rows(&self) -> usize {
        self.shape()[0]
    }

    fn num_cols(&self) -> usize {
        self.shape()[1]
    }

    fn mut_rows(&mut self, f: impl Fn(usize, &mut [f32]) + Sync) {
        // Simple, sequential implementation using mut_row.
        for row_idx in 0..self.num_rows() {
            self.mut_row([row_idx, 0], |row| f(row_idx, row));
        }
    }
}
