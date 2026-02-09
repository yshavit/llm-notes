pub mod gelu;
mod matmul;
pub mod norm;
mod sample;
pub mod softmax;
mod tensor;
mod trait_adapter;

use crate::tensor::LayerNorm;
pub use sample::LogitSampler;
pub use softmax::*;
pub use tensor::*;

/// CPU-based tensor backend implementation
pub struct CpuBackend;

impl crate::tensor::TensorBackend for CpuBackend {
    type Tensor<const R: usize> = CpuTensor<R>;
    type LayerNorm = crate::cputensor::norm::CpuLayerNorm;
}
