use crate::tensor::Shape;
use std::fmt::{Debug, Formatter};

pub struct Tensor<const R: usize> {
    data: Vec<f32>,
    shape: Shape<R>,
    strides: Shape<R>,
}

pub type Vector = Tensor<1>;
pub type Matrix = Tensor<2>;

impl<const R: usize> Tensor<R> {
    pub fn new(shape: Shape<R>) -> Self {
        assert_ne!(R, 0, "0-tensors are not allowed");
        let mut strides = [0; R];
        // Work backwards from the last dimension: The last dimension is contiguous by default (stride = 1), and then
        // each dimension back needs to have a stride-size for all the dimensions before it.
        let mut stride = 1;
        for i in (0..R).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        Self {
            data: vec![0.0; shape.num_elements()],
            shape,
            strides: Shape::new(strides),
        }
    }

    pub fn shape(&self) -> Shape<R> {
        self.shape
    }

    fn data_offset(&self, indices: [usize; R]) -> usize {
        let mut offset = 0;
        for i in 0..R {
            assert!(
                indices[i] < self.shape[i],
                "index out of range: can't get {indices:?} on {} tensor",
                self.shape
            );
            offset += self.strides[i] * indices[i]
        }
        offset
    }

    pub fn get(&self, indices: [usize; R]) -> f32 {
        self.data[self.data_offset(indices)]
    }

    pub fn transposed(mut self, dim0: usize, dim1: usize) -> Self {
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
        self
    }

    /// Sets all or part of a row's values. The first `R-1` indices specify an offset into the tensor, and the last
    /// index specifies an offset into the row. That offset + `values.len()` must be less than the last index's
    /// dimensionality.
    pub fn set_row(&mut self, indices: [usize; R], values: &[f32]) {
        let write_start = self.data_offset(indices);
        let row_offset = indices[R - 1];
        assert!(
            values.len() <= self.shape[R - 1] - row_offset,
            "can't write to index {:?} of {} matrix with slice length {}",
            indices,
            self.shape,
            values.len()
        );
        if self.strides[R - 1] == 1 {
            // Row-major format; we can just memcpy
            self.data[write_start..write_start + self.shape[R - 1]].copy_from_slice(values);
        } else {
            let stride: usize = (0..R - 1).map(|i| self.strides[i] * self.shape[i]).sum();
            let mut offset = write_start;
            for val in values {
                self.data[offset] = *val;
                offset += stride;
            }
        }
    }
}

impl<const R: usize> Debug for Tensor<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ", self.shape)?;
        match R {
            1 => write!(f, "vector"),
            2 => write!(f, "matrix"),
            _ => write!(f, "tensor"),
        }
    }
}

struct Pretty<'a, const R: usize>(&'a Tensor<R>);

impl<'a> Debug for Pretty<'a, 2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tensor = self.0;
        // Turn all the values into equal-length, padded strings
        let mut cell_strings = Vec::with_capacity(tensor.num_rows() * tensor.num_cols());
        let mut max_cell_len = 0;
        for row in 0..tensor.num_rows() {
            for col in 0..tensor.num_cols() {
                let val_string = tensor.get([row, col]).to_string();
                max_cell_len = max_cell_len.max(val_string.len());
                cell_strings.push(val_string);
            }
        }
        cell_strings
            .iter_mut()
            .for_each(|s| *s = format!("{:>width$}", s, width = max_cell_len));

        // Now write them all. The cell_strings is already in row-major order, so we can just keep track of newlines.
        write!(f, ":")?;
        let mut line_tracker = tensor.num_cols(); // start a newline right away
        for s in cell_strings {
            if line_tracker >= tensor.num_cols() {
                write!(f, "\n|")?;
                line_tracker = 0;
            }
            write!(f, " {s} |")?;
            line_tracker += 1;
        }

        Ok(())
    }
}

impl Tensor<2> {
    pub fn new_matrix(num_rows: usize, num_columns: usize) -> Self {
        Self::new(Shape::new([num_rows, num_columns]))
    }

    pub fn t(self) -> Self {
        self.transposed(0, 1)
    }

    pub fn num_rows(&self) -> usize {
        self.shape[0]
    }

    pub fn num_cols(&self) -> usize {
        self.shape[1]
    }
}

impl<const R: usize> PartialEq for Tensor<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }

        // Fast path: same strides means same layout, so we can do just a simple == on the data
        if self.strides == other.strides {
            return self.data == other.data;
        }

        // Slow path: different strides, so we have to compare element by element
        for indices in self.shape.iter_indices() {
            if self.get(indices) != other.get(indices) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;

    /// Smoke test of the various shapes of things.
    #[test]
    fn matrix_data_shape() {
        let m = Matrix::new_matrix(3, 4);
        assert_eq!(m.num_rows(), 3);
        assert_eq!(m.num_cols(), 4);
        assert_eq!(m.shape(), Shape::new([3, 4]));

        check_row(&m, 0, [0., 0., 0., 0.]);
        check_row(&m, 1, [0., 0., 0., 0.]);
        check_row(&m, 2, [0., 0., 0., 0.]);
        expect_panic(|| m.get([3, 0]));
    }

    #[test]
    fn data_round_trip() {
        let mut m = Matrix::new_matrix(3, 4);

        let values = &[1., 2., 3., 4.];
        m.set_row([0, 0], values);
        let values = &[5., 6., 7., 8.];
        m.set_row([1, 0], values);
        let values = &[9., 10., 11., 12.];
        m.set_row([2, 0], values);

        check_row(&m, 0, [1., 2., 3., 4.]);
        check_row(&m, 1, [5., 6., 7., 8.]);
        check_row(&m, 2, [9., 10., 11., 12.]);
    }

    #[test]
    #[should_panic = "can't write to index [0, 0] of 3x4 matrix with slice length 6"]
    fn row_mut_set_all_bounds() {
        let mut m = Matrix::new_matrix(3, 4);
        let values = &[1., 2., 3., 4., 5., 6.];
        m.set_row([0, 0], values);
    }

    #[test]
    fn transposition() {
        let mut m = Matrix::new_matrix(3, 4);

        let values = &[1., 2., 3., 4.];
        m.set_row([0, 0], values);
        let values = &[5., 6., 7., 8.];
        m.set_row([1, 0], values);
        let values = &[9., 10., 11., 12.];
        m.set_row([2, 0], values);

        let transposed = m.t();
        assert_eq!(transposed.shape(), Shape::new([4, 3]));

        check_row(&transposed, 0, [1., 5., 9.]);
        check_row(&transposed, 1, [2., 6., 10.]);
        check_row(&transposed, 2, [3., 7., 11.]);
        check_row(&transposed, 3, [4., 8., 12.]);
        expect_panic(|| transposed.get([4, 0]));

        // quick sanity check on double-transposition
        let double_transposed = transposed.t();
        check_row(&double_transposed, 0, [1., 2., 3., 4.]);
    }

    #[test]
    fn transposed_set_row() {
        let mut m = Matrix::new_matrix(3, 4);

        let mut transposed = m.t();
        let values = &[1., 2., 3.];
        transposed.set_row([1, 0], values);

        check_row(&transposed, 0, [0., 0., 0.]);
        check_row(&transposed, 1, [1., 2., 3.]);
        check_row(&transposed, 2, [0., 0., 0.]);
        check_row(&transposed, 3, [0., 0., 0.]);
    }

    fn check_row<const N: usize>(m: &Matrix, row: usize, expected: [f32; N]) {
        let actual: Vec<_> = (0..N).map(|i| m.get([row, i])).collect();
        let expected = Vec::from(expected);
        assert_eq!(actual, expected);

        expect_panic(|| m.get([row, N]))
    }

    fn expect_panic<X>(f: impl FnOnce() -> X + panic::UnwindSafe) {
        assert!(panic::catch_unwind(f).is_err());
    }
}
