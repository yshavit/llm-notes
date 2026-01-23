use crate::tensor::matrix::{Matrix, MatrixMut};
use crate::tensor::vector::{Vector, VectorMut};

pub fn matmul(a: impl Matrix, b: impl Matrix, out: &mut impl MatrixMut) {
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

        // Conceptually, what we want is just a bunch of dot products:
        //
        //     for col_idx in 0..b.num_cols() {
        //         row_vals[col_idx] = Self::dot(a.row(row_idx), b.col(col_idx));
        //     }
        //
        // This performs badly in practice, because the b.col(col_idx) iteration has to jump from row to row. Each
        // one of those jumps will be a cache miss: so basically, the dot product will have as many cache misses as
        // it has elements.
        //
        // Instead, we're going to go row by row on both matrices, calculating only a portion of the dot product at
        // a time. We start with the first row of a. We fetch the first item of that a row, and then multiply it
        // with each value in the b row. Then we fetch the second value of a, and again multiply it with each value
        // in the b row, adding those terms to our first value -- and so on. Note that while this is a triple-nested
        // loop (each row of a will have to iterate over b's row N times, where N is a's column count), a and b are
        // both always read in row-sequence order. This is very cache-friendly, and makes it easier for the L1/L2/L3
        // cache lines to predict our reads.
        for (k, a_val) in a.row(row_idx).iter().enumerate() {
            for (col_idx, b_val) in b.row(k).iter().enumerate() {
                row_vals[col_idx] += a_val * b_val;
            }
        }
        out.row_mut(row_idx).set_all(&row_vals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod matrix {
        use super::*;

        #[test]
        fn check_matmul() {
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
            matmul(a, b, &mut out);
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
            matmul(a, b, &mut out);
        }

        #[test]
        #[should_panic = "can't multiply (2x3) and (3x4) into (4x2)"]
        fn matmul_output_mismatch() {
            let a = array_matrix::<2, 3>();
            let b = array_matrix::<3, 4>();
            let mut out = array_matrix::<4, 2>();
            matmul(a, b, &mut out);
        }
    }

    fn array_matrix<const R: usize, const C: usize>() -> [[f32; C]; R] {
        [[0.; C]; R]
    }

    impl<const N: usize> Vector for [f32; N] {
        fn len(&self) -> usize {
            N
        }

        fn get(&self, idx: usize) -> f32 {
            self[idx]
        }
    }

    impl<const N: usize> Vector for &mut [f32; N] {
        fn len(&self) -> usize {
            N
        }

        fn get(&self, idx: usize) -> f32 {
            self[idx]
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
