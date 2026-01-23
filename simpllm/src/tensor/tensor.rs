use crate::tensor::Shape;
use crate::tensor::vector::Vector;

type TMatrix = Tensor<2>;
type TVector = Tensor<1>;

struct Tensor<const R: usize> {
    data: Vec<f32>,
    shape: Shape<R>,
    strides: Shape<R>,
}

impl<const R: usize> Tensor<R> {
    pub fn new(shape: Shape<R>) -> Self {
        let mut strides = [0; R];
        let mut stride = 1;

        // Work backwards from the last dimension: The last dimension is contiguous by default (stride = 1), and then
        // each dimension back needs to have a stride-size for all the dimensions before it.
        for i in (0..R).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        let strides = Shape::from(strides);

        Self {
            data: vec![0.0; shape.num_elements()],
            shape,
            strides,
        }
    }

    pub fn shape(&self) -> Shape<R> {
        self.shape
    }

    pub fn get(&self, indices: [usize; R]) -> f32 {
        indices.iter().zip(self.shape.iter()).for_each(|(idx, dim)| {
            if idx >= dim {
                panic!("index out of range: can't get {indices:?} on {} tensosr", self.shape);
            }
        });
        let flat_index = indices
            .iter()
            .zip(self.strides.iter())
            .map(|(idx, stride)| idx * stride)
            .sum::<usize>();

        self.data[flat_index]
    }

    fn transposed(self, dim0: usize, dim1: usize) -> Self {
        Self {
            data: self.data,
            shape: self.shape.swapped(dim0, dim1),
            strides: self.strides.swapped(dim0, dim1),
        }
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

    pub fn set_row(&mut self, row: usize, values: &[f32]) {
        let row_len = self.num_cols();
        if values.len() != row_len {
            panic!(
                "can't write {} values to row {} of {} matrix",
                values.len(),
                row,
                self.shape
            )
        }
        if row >= self.num_rows() {
            panic!("can't write to row {} of {} matrix", row, self.shape)
        }
        let start = row * row_len;
        self.data[start..start + row_len].copy_from_slice(values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;

    /// Smoke test of the various shapes of things.
    #[test]
    fn matrix_data_shape() {
        let m = TMatrix::new_matrix(3, 4);
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
        let mut m = TMatrix::new_matrix(3, 4);

        m.set_row(0, &[1., 2., 3., 4.]);
        m.set_row(1, &[5., 6., 7., 8.]);
        m.set_row(2, &[9., 10., 11., 12.]);

        check_row(&m, 0, [1., 2., 3., 4.]);
        check_row(&m, 1, [5., 6., 7., 8.]);
        check_row(&m, 2, [9., 10., 11., 12.]);
    }

    #[test]
    #[should_panic = "can't write 6 values to row 0 of 3x4 matrix"]
    fn row_mut_set_all_bounds() {
        let mut m = TMatrix::new_matrix(3, 4);
        m.set_row(0, &[1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn transposition() {
        let mut m = TMatrix::new_matrix(3, 4);

        m.set_row(0, &[1., 2., 3., 4.]);
        m.set_row(1, &[5., 6., 7., 8.]);
        m.set_row(2, &[9., 10., 11., 12.]);

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

    fn check_row<const N: usize>(m: &TMatrix, row: usize, expected: [f32; N]) {
        let actual: Vec<_> = (0..N).map(|i| m.get([row, i])).collect();
        let expected = Vec::from(expected);
        assert_eq!(actual, expected);

        expect_panic(|| m.get([row, N]))
    }

    fn expect_panic<X>(f: impl FnOnce() -> X + panic::UnwindSafe) {
        assert!(panic::catch_unwind(f).is_err());
    }
}
