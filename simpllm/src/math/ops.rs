use crate::math::matrix::{Matrix, MatrixMut};
use crate::math::vector::{Vector, VectorMut};

pub trait MatrixOps {
    fn matmul(a: impl Matrix, b: impl Matrix, out: &mut impl MatrixMut);
}

pub trait VectorOps {
    fn dot(a: impl Vector, b: impl Vector) -> f32 {
        if a.len() != b.len() {
            panic!("can't dot vectors of different lengths: {} != {}", a.len(), b.len(),);
        }
        let mut result = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            result += a_val * b_val;
        }
        result
    }
}

pub struct CpuOps;

impl VectorOps for CpuOps {}

impl MatrixOps for CpuOps {
    fn matmul(a: impl Matrix, b: impl Matrix, out: &mut impl MatrixMut) {
        if a.num_cols() != b.num_rows() {
            panic!(
                "can't multiply ({}) and ({}) [into ({})]",
                a.shape(),
                b.shape(),
                out.shape(),
            );
        }
        if a.num_rows() != out.num_rows() || b.num_cols() != out.num_cols() {
            panic!(
                "can't multiply ({}) and ({}) into ({})",
                a.shape(),
                b.shape(),
                out.shape(),
            );
        }

        let mut row_vals = vec![0.0; a.num_cols()];

        for row_idx in 0..a.num_rows() {
            row_vals.fill(0.0);
            for col_idx in 0..b.num_cols() {
                row_vals[col_idx] = Self::dot(a.row(row_idx), b.col(col_idx));
            }
            out.row_mut(row_idx).set_all(&row_vals);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod vector {
        use super::*;

        #[test]
        fn dot_product() {
            let a = [1.0, 2.0, 3.0];
            let b = [4.0, 5.0, 6.0];

            assert_eq!(CpuOps::dot(a, b), (1. * 4.) + (2. * 5.) + (3. * 6.));
        }

        #[test]
        #[should_panic = "can't dot vectors of different lengths: 2 != 3"]
        fn dot_product_length_mismatch() {
            let a = [1.0, 2.0];
            let b = [3.0, 4.0, 5.0];

            CpuOps::dot(a, b);
        }
    }

    mod matrix {
        use super::*;

        #[test]
        fn matmul() {
            let a = [
                // comment, so that rustfmt doesn't collapse this to a single line
                [1., 2.],
                [3., 4.],
                [5., 6.],
            ];
            let b = [
                //
                [7., 8.],
                [9., 10.],
            ];
            let mut out = [[0.0; 2]; 3];
            CpuOps::matmul(a, b, &mut out);
            assert_eq!(
                out,
                [
                    [(1. * 7.) + (2. * 9.), (1. * 8.) + (2. * 10.)],
                    [(3. * 7.) + (4. * 9.), (3. * 8.) + (4. * 10.)],
                    [(5. * 7.) + (6. * 9.), (5. * 8.) + (6. * 10.)],
                ]
            );
        }

        #[test]
        #[should_panic = "can't multiply (2x3) and (2x3) [into (2x3)]"]
        fn matmul_a_b_mismatch() {
            let a = array_matrix::<2, 3>();
            let b = array_matrix::<2, 3>();
            let mut out = array_matrix::<2, 3>();
            CpuOps::matmul(a, b, &mut out);
        }

        #[test]
        #[should_panic = "can't multiply (2x3) and (3x4) into (4x2)"]
        fn matmul_output_mismatch() {
            let a = array_matrix::<2, 3>();
            let b = array_matrix::<3, 4>();
            let mut out = array_matrix::<4, 2>();
            CpuOps::matmul(a, b, &mut out);
        }
    }

    fn array_matrix<const R: usize, const C: usize>() -> [[f32; C]; R] {
        [[0.; C]; R]
    }

    impl<const N: usize> Vector for [f32; N] {
        fn len(&self) -> usize {
            N
        }

        fn iter(&self) -> impl Iterator<Item = f32> {
            self.as_slice().iter().copied()
        }
    }

    impl<const N: usize> Vector for &mut [f32; N] {
        fn len(&self) -> usize {
            N
        }

        fn iter(&self) -> impl Iterator<Item = f32> {
            self.as_slice().iter().copied()
        }
    }

    impl<const N: usize> VectorMut for &mut [f32; N] {
        fn set(&mut self, idx: usize, value: f32) {
            self[idx] = value;
        }
    }

    impl<const R: usize, const C: usize> Matrix for [[f32; C]; R] {
        fn num_rows(&self) -> usize {
            R
        }

        fn num_cols(&self) -> usize {
            C
        }

        fn row(&self, row: usize) -> impl Vector {
            let mut res = [0.0; C];
            res.copy_from_slice(&self[row]);
            res
        }

        fn col(&self, col: usize) -> impl Vector {
            let mut res = [0.0; R];
            for row in 0..self.num_rows() {
                res[row] = self[row][col];
            }
            res
        }
    }

    impl<const R: usize, const C: usize> MatrixMut for [[f32; C]; R] {
        fn row_mut(&mut self, row: usize) -> impl VectorMut {
            &mut self[row]
        }
    }
}
