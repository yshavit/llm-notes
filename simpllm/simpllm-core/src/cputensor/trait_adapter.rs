use crate::cputensor::matmul::matmul_batched;
use crate::tensor::{Shape, TensorBackend, TensorSlice};
use std::borrow::Cow;

impl TensorSlice for [f32] {
    fn flat_f32(&self) -> Cow<'_, [f32]> {
        Cow::from(self)
    }
}

impl<const R: usize> crate::tensor::Tensor<R> for super::CpuTensor<R> {
    type Backend = crate::cputensor::CpuBackend;
    type Slice = [f32];

    fn zeros(shape: impl Into<Shape<R>>) -> Self {
        super::CpuTensor::new(shape)
    }

    fn shape(&self) -> Shape<R> {
        self.shape()
    }

    fn reset_values(&mut self, values: &[f32]) {
        self.reset_values(values)
    }

    fn reshape<const R2: usize>(self, new_shape: impl Into<Shape<R2>>) -> <Self::Backend as TensorBackend>::Tensor<R2> {
        self.reshape(new_shape)
    }

    fn split<const S: usize>(self, dim: usize) -> [<Self::Backend as TensorBackend>::Tensor<R>; S] {
        self.split(dim)
    }

    fn transposed(self, dim0: usize, dim1: usize) -> Self {
        self.transposed(dim0, dim1)
    }

    fn contiguous(self) -> Self {
        self.contiguous()
    }

    fn matmul(&self, other: &Self) -> Self {
        matmul_batched(self, other)
    }

    fn slice_row<X>(&self, indices: [usize; R], f: impl FnOnce(&Self::Slice) -> X) -> X {
        self.with_row(indices, f)
    }

    fn set_slice(&mut self, indices: [usize; R], values: &Self::Slice) {
        self.set_row(indices, values)
    }

    fn flat_f32(&self) -> Cow<'_, [f32]> {
        self.flat_f32()
    }

    fn gelu(self) -> Self {
        self.gelu()
    }

    fn softmax(self) -> Self {
        self.softmax()
    }

    fn multiply_scalar(&mut self, factor: f32) {
        self.multiply_scalar(factor)
    }

    fn add<const R2: usize>(self, other: &<Self::Backend as TensorBackend>::Tensor<R2>) -> Self {
        self.add_tensor(other)
    }
}
