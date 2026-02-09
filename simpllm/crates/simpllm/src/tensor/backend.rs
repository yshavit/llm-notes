use super::Tensor;

/// Abstraction over different tensor backend implementations (CPU, GPU, etc.)
pub trait TensorBackend: Sized {
    type Tensor<const R: usize>: Tensor<R, Backend = Self>;

    /// Create a new matrix (rank-2 tensor)
    fn new_matrix(rows: usize, cols: usize) -> Self::Tensor<2> {
        Self::Tensor::new([rows, cols])
    }

    /// Create a new vector (rank-1 tensor)
    fn new_vector(size: usize) -> Self::Tensor<1> {
        Self::Tensor::new([size])
    }
}

/// Convenience type alias for a vector from a backend
pub type Vector<B> = <B as TensorBackend>::Tensor<1>;

/// Convenience type alias for a matrix from a backend
pub type Matrix<B> = <B as TensorBackend>::Tensor<2>;
