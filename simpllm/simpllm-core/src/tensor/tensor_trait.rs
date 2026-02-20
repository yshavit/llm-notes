use crate::tensor::{Shape, TensorBackend};
use std::borrow::Cow;
use std::fmt::Debug;

pub trait Tensor<const R: usize>: Sized + Clone + Send + Sync + Debug {
    type Backend: TensorBackend;
    type Slice: TensorSlice + ?Sized;

    fn from_row_major(shape: impl Into<Shape<R>>, data: &[f32]) -> Self;

    fn zeros(shape: impl Into<Shape<R>>) -> Self {
        let shape: Shape<R> = shape.into();
        let data = vec![0.0; shape.num_elements()];
        Self::from_row_major(shape, &data)
    }

    fn shape(&self) -> Shape<R>;

    fn reshape<const R2: usize>(self, new_shape: impl Into<Shape<R2>>) -> <Self::Backend as TensorBackend>::Tensor<R2>;
    fn split<const S: usize>(self, dim: usize) -> [<Self::Backend as TensorBackend>::Tensor<R>; S];
    fn transposed(self, dim0: usize, dim1: usize) -> Self;
    fn contiguous(self) -> Self;

    fn flat_f32(&self) -> Cow<'_, [f32]>;
    fn slice_row<X>(&self, indices: [usize; R], f: impl FnOnce(&Self::Slice) -> X) -> X;
    fn set_slice(&mut self, indices: [usize; R], values: &Self::Slice);

    fn gelu(self) -> Self;
    fn softmax(self) -> Self;

    fn matmul(&self, other: &Self) -> Self;
    fn multiply_scalar(&mut self, factor: f32);
    fn add<const R2: usize>(self, other: &<Self::Backend as TensorBackend>::Tensor<R2>) -> Self;
}

pub trait TensorSlice {
    fn flat_f32(&self) -> Cow<'_, [f32]>;
}

pub trait Tensor2D: Tensor<2> {
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
}

impl<T: Tensor<2>> Tensor2D for T {
    fn num_rows(&self) -> usize {
        self.shape()[0]
    }

    fn num_cols(&self) -> usize {
        self.shape()[1]
    }
}
