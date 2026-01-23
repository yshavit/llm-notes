use crate::tensor::shape::Shape;
use crate::tensor::vector::{Vector, VectorMut};

pub trait Matrix: Sized {
    fn shape(&self) -> Shape<2>;
    fn row(&self, row: usize) -> impl Vector;
    fn transpose(&self) -> impl Matrix {
        TransposedMatrix { underlying: self }
    }

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

// A simple implementation of a 2-d matrix.
pub struct MatrixData {
    shape: Shape<2>,
    data: Vec<f32>,
}

impl MatrixData {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            shape: Shape::new([num_rows, num_cols]),
            data: vec![0.0; num_rows * num_cols],
        }
    }
}

impl Matrix for MatrixData {
    fn shape(&self) -> Shape<2> {
        self.shape
    }

    fn row(&self, row: usize) -> impl Vector {
        if row >= self.num_rows() {
            panic!("can't get row {} from {} matrix", row, self.shape());
        }
        MatrixDataRow { matrix: self, row }
    }
}

impl MatrixMut for MatrixData {
    fn row_mut(&mut self, row: usize) -> impl VectorMut {
        if row >= self.num_rows() {
            panic!("can't get row {} from {} matrix", row, self.shape());
        }
        MatrixDataRowMut { matrix: self, row }
    }
}

struct MatrixDataRow<'a> {
    matrix: &'a MatrixData,
    row: usize,
}

impl MatrixDataRow<'_> {
    fn row_iter(m: &MatrixData, row: usize) -> impl Iterator<Item = f32> {
        let start = row * m.num_cols();
        m.data[start..start + m.num_cols()].iter().copied()
    }
}

impl<'a> Vector for MatrixDataRow<'a> {
    fn len(&self) -> usize {
        self.matrix.num_cols()
    }

    fn get(&self, idx: usize) -> f32 {
        self.matrix.data[self.row * self.matrix.num_cols() + idx]
    }

    fn iter(&self) -> impl Iterator<Item = f32> {
        Self::row_iter(self.matrix, self.row)
    }
}

struct MatrixDataRowMut<'a> {
    matrix: &'a mut MatrixData,
    row: usize,
}

impl<'a> Vector for MatrixDataRowMut<'a> {
    fn len(&self) -> usize {
        self.matrix.num_cols()
    }

    fn get(&self, idx: usize) -> f32 {
        self.matrix.data[self.row * self.matrix.num_cols() + idx]
    }

    fn iter(&self) -> impl Iterator<Item = f32> {
        MatrixDataRow::row_iter(self.matrix, self.row)
    }
}

impl<'a> VectorMut for MatrixDataRowMut<'a> {
    fn set(&mut self, col: usize, value: f32) {
        if col >= self.len() {
            panic!(
                "can't set value at index ({}, {}) in {} matrix",
                self.row,
                col,
                self.matrix.shape()
            );
        }
        let num_cols = self.matrix.num_cols();
        self.matrix.data[self.row * num_cols + col] = value;
    }

    fn set_all(&mut self, values: &[f32]) {
        let len = self.len();
        if values.len() != len {
            panic!(
                "can't set {} values in row {} of {} matrix",
                values.len(),
                self.row,
                self.matrix.shape()
            );
        }
        let start = self.row * self.matrix.num_cols();
        self.matrix.data[start..start + len].copy_from_slice(values);
    }
}

struct TransposedMatrix<'a, M> {
    underlying: &'a M,
}

impl<'a, M: Matrix> Matrix for TransposedMatrix<'a, M> {
    fn shape(&self) -> Shape<2> {
        let underlying_shape = self.underlying.shape();
        Shape::new([underlying_shape[1], underlying_shape[0]])
    }

    fn row(&self, row: usize) -> impl Vector {
        todo!() as MatrixDataRow
    }

    fn transpose(&self) -> impl Matrix {
        self.underlying
    }
}

struct MatrixDataCol<'a> {
    matrix: &'a MatrixData,
    col: usize,
}

impl<'a> Vector for MatrixDataCol<'a> {
    fn len(&self) -> usize {
        self.matrix.num_rows()
    }

    fn get(&self, idx: usize) -> f32 {
        self.matrix.data[idx * self.matrix.num_cols() + self.col]
    }
}

/// Lets `&impl Matrix` be `Matrix`, so that [`TransposedMatrix::transpose`] can just return its underlying reference.
mod transposing {
    use crate::tensor::vector::Vector;
    use crate::tensor::{Matrix, Shape};

    impl<M: Matrix> Matrix for &M {
        fn shape(&self) -> Shape<2> {
            M::shape(self)
        }

        fn row(&self, row: usize) -> impl Vector {
            M::row(self, row)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;

    /// Smoke test of the various shapes of things.
    #[test]
    fn matrix_data_shape() {
        let m = MatrixData::new(3, 4);
        assert_eq!(m.num_rows(), 3);
        assert_eq!(m.num_cols(), 4);
        assert_eq!(m.shape(), Shape::new([3, 4]));

        check_vector(m.row(0), &[0., 0., 0., 0.]);
        check_vector(m.row(1), &[0., 0., 0., 0.]);
        check_vector(m.row(2), &[0., 0., 0., 0.]);
        expect_panic(|| m.row(3));
    }

    #[test]
    fn row_iter() {
        let mut m = MatrixData::new(3, 4);

        m.row_mut(0).set_all(&[1., 2., 3., 4.]);
        m.row_mut(1).set_all(&[5., 6., 7., 8.]);
        m.row_mut(2).set_all(&[9., 10., 11., 12.]);

        check_vector(m.row(0), &[1., 2., 3., 4.]);
        check_vector(m.row(1), &[5., 6., 7., 8.]);
        check_vector(m.row(2), &[9., 10., 11., 12.]);
    }

    #[test]
    fn col_iter() {
        let mut m = MatrixData::new(3, 4);

        m.row_mut(0).set_all(&[1., 2., 3., 4.]);
        m.row_mut(1).set_all(&[5., 6., 7., 8.]);
        m.row_mut(2).set_all(&[9., 10., 11., 12.]);
    }

    /// Checks that `row_mut()`'s implementation also performs well as a non-mut vector
    #[test]
    fn row_mut_as_vector() {
        let mut m = MatrixData::new(3, 4);

        check_vector(m.row_mut(0), &[0., 0., 0., 0.]);
        m.row_mut(0).set_all(&[1., 2., 3., 4.]);
        check_vector(m.row_mut(0), &[1., 2., 3., 4.]);
    }

    #[test]
    #[should_panic = "can't get row 3 from 3x4 matrix"]
    fn row_mut_bounds() {
        let mut m = MatrixData::new(3, 4);
        m.row_mut(3);
    }

    #[test]
    #[should_panic = "can't set value at index (0, 10) in 3x4 matrix"]
    fn row_mut_idx_bounds() {
        let mut m = MatrixData::new(3, 4);
        m.row_mut(0).set(10, 2.0);
    }

    #[test]
    #[should_panic = "can't set 6 values in row 0 of 3x4 matrix"]
    fn row_mut_set_all_bounds() {
        let mut m = MatrixData::new(3, 4);
        m.row_mut(0).set_all(&[1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn transposition() {
        let mut m = MatrixData::new(3, 4);

        m.row_mut(0).set_all(&[1., 2., 3., 4.]);
        m.row_mut(1).set_all(&[5., 6., 7., 8.]);
        m.row_mut(2).set_all(&[9., 10., 11., 12.]);

        let transposed = m.transpose();
        assert_eq!(transposed.shape(), Shape::new([4, 3]));

        check_vector(transposed.row(0), &[1., 5., 9.]);
        check_vector(transposed.row(1), &[2., 6., 10.]);
        check_vector(transposed.row(2), &[3., 7., 11.]);
        check_vector(transposed.row(3), &[4., 8., 12.]);
        expect_panic(|| transposed.row(4));

        // quick sanity check on double-transposition
        let double_transposed = transposed.transpose();
        check_vector(double_transposed.row(0), &[1., 2., 3., 4.]);
    }

    fn check_vector(v: impl Vector, expected: &[f32]) {
        assert_eq!(v.len(), expected.len());
        assert_eq!(v.iter().collect::<Vec<_>>(), expected);
    }

    fn expect_panic<X>(f: impl FnOnce() -> X + panic::UnwindSafe) {
        assert!(panic::catch_unwind(f).is_err());
    }
}
