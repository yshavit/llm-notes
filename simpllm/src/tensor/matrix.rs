use crate::tensor::shape::Shape;
use crate::tensor::vector::VectorMut;

pub trait Matrix: Sized {
    fn shape(&self) -> Shape<2>;
    fn get(&self, row: usize, col: usize) -> f32;

    fn num_rows(&self) -> usize {
        self.shape()[0]
    }

    fn num_cols(&self) -> usize {
        self.shape()[1]
    }
}

pub trait MatrixMut: Matrix {
    fn row_mut(&mut self, row: usize) -> impl VectorMut;
}
