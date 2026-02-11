use crate::tensor::{Matrix, TensorBackend, Vector};

pub trait LayerNorm {
    type B: TensorBackend;

    fn new(scale: Vector<Self::B>, bias: Vector<Self::B>, epsilon: f32) -> Self;

    fn apply(&self, input: &Matrix<Self::B>) -> Matrix<Self::B>;
}
