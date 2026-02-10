use super::{LayerNorm, Tensor};

/// Abstraction over different tensor backend implementations (CPU, GPU, etc.)
pub trait TensorBackend: Sized {
    type Tensor<const R: usize>: Tensor<R, Backend = Self>;
    type LayerNorm: LayerNorm<B = Self>;

    fn lower_triangle(n: usize) -> Self::Tensor<2>;

    fn new_matrix(rows: usize, cols: usize) -> Self::Tensor<2> {
        Self::Tensor::new([rows, cols])
    }

    fn new_vector(size: usize) -> Self::Tensor<1> {
        Self::Tensor::new([size])
    }
}

/// Convenience type
pub type Vector<B> = <B as TensorBackend>::Tensor<1>;

/// Convenience type
pub type Matrix<B> = <B as TensorBackend>::Tensor<2>;
