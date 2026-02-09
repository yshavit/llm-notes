pub mod gelu;
mod matmul;
mod sample;
pub mod softmax;
mod tensor;
mod trait_adapter;

pub use sample::LogitSampler;
pub use softmax::*;
pub use tensor::*;

/// CPU-based tensor backend implementation
pub struct CpuBackend;

impl crate::tensor::TensorBackend for CpuBackend {
    type Tensor<const R: usize> = CpuTensor<R>;
}
