pub mod gelu;
mod matmul;
pub mod norm;
pub mod softmax;
mod tensor;
mod trait_adapter;

pub use crate::llm::sample::LogitSampler;
use crate::tensor::Tensor;
pub use softmax::*;
pub use tensor::*;

/// CPU-based tensor backend implementation
pub struct CpuBackend;

impl CpuBackend {
    pub fn pretty_print_matrix(matrix: &impl Tensor<2>) -> String {
        let matrix_data = matrix.flat_f32().into_owned();
        let matrix_cpu = CpuTensor::new_with_data(matrix.shape(), matrix_data);
        format!("{matrix_cpu}")
    }
}

impl crate::tensor::TensorBackend for CpuBackend {
    type Tensor<const R: usize> = CpuTensor<R>;
    type LayerNorm = norm::CpuLayerNorm;

    fn lower_triangle(n: usize) -> Self::Tensor<2> {
        let mut data = vec![0.0; n * n];
        data.chunks_exact_mut(n).enumerate().for_each(|(i, row_data)| {
            row_data.iter_mut().skip(i + 1).for_each(|v| *v = f32::NEG_INFINITY);
        });
        let mut triangle_matrix = CpuTensor::new([n, n]);
        triangle_matrix.reset_values(&data);
        triangle_matrix
    }
}
