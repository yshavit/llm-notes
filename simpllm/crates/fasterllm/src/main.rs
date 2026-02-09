use candle_core::{DType, Device};
use simpllm::tensor::{Shape, Tensor, TensorBackend};
use std::borrow::Cow;
use std::sync::LazyLock;

fn main() {
    println!("Hello, world!");
}

static CUDA: LazyLock<Device> = LazyLock::new(|| Device::new_cuda(0).expect("couldn't initialize CUDA"));

struct CandleBackend;

impl TensorBackend for CandleBackend {
    type Tensor<const R: usize> = ();
}

#[derive(Clone)]
struct CandleTensor<const R: usize> {
    c_tensor: candle_core::Tensor,
}

impl<const R: usize> Tensor<R> for CandleTensor<R> {
    type Backend = CandleBackend;

    fn new(shape: impl Into<Shape<R>>) -> Self {
        let shape: candle_core::Shape = shape.into().into();
        let t_tensor = candle_core::Tensor::zeros(shape, DType::F32, &CUDA).expect("couldn't create candle tensor");
        Self { c_tensor }
    }

    fn shape(&self) -> Shape<R> {
        self.c_tensor.shape().into()
    }

    fn reset_values(&mut self, values: &[f32]) {
        self.c_tensor = candle_core::Tensor::from_slice(values, self.c_tensor.shape(), self.c_tensor.device())
            .expect("couldn't create new tensor");
    }

    fn reshape<const R2: usize>(self, new_shape: impl Into<Shape<R2>>) -> <Self::Backend as TensorBackend>::Tensor<R2> {
        todo!()
    }

    fn split<const S: usize>(self, dim: usize) -> [<Self::Backend as TensorBackend>::Tensor<R>; S] {
        todo!()
    }

    fn transposed(self, dim0: usize, dim1: usize) -> Self {
        todo!()
    }

    fn contiguous(self) -> Self {
        todo!()
    }

    fn matmul(&self, other: &Self) -> Self {
        todo!()
    }

    fn get(&self, indices: [usize; R]) -> f32 {
        todo!()
    }

    fn with_row<X>(&self, indices: [usize; R], f: impl FnOnce(&[f32]) -> X) -> X {
        todo!()
    }

    fn mut_row<X>(&mut self, indices: [usize; R], f: impl FnOnce(&mut [f32]) -> X) -> X {
        todo!()
    }

    fn flat_f32(&self) -> Cow<'_, [f32]> {
        todo!()
    }

    fn gelu(self) -> Self {
        todo!()
    }

    fn softmax(self) -> Self {
        todo!()
    }

    fn multiply_scalar(&mut self, factor: f32) {
        todo!()
    }

    fn add<const R2: usize>(self, other: <Self::Backend as TensorBackend>::Tensor<R2>) -> Self {
        todo!()
    }

    fn set_row(&mut self, indices: [usize; R], values: &[f32]) {
        todo!()
    }
}
