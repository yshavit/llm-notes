use crate::tensor::tensor::{MatrixView, MatrixViewMut, Tensor};

pub fn matmul_batched(a: &Tensor<3>, b: &Tensor<2>, out: &mut Tensor<3>) {
    assert_eq!(
        a.shape()[0],
        out.shape()[0],
        "batch dimensions don't match: can't multiply ({}) and ({}) into ({})",
        a.shape(),
        b.shape(),
        out.shape()
    );
    // We don't need to check the non-batch dimensions: matmul will do that for us

    let num_batches = a.shape()[0];
    for batch in 0..num_batches {
        let batch_indices = [batch, 0, 0];
        let a_matrix = a.matrix_slice(batch_indices);
        let out_matrix = out.matrix_slice_mut(batch_indices);
        matmul(a_matrix, b, out_matrix);
    }
}

pub fn matmul<'a, const A: usize, const B: usize, const C: usize>(
    a: impl Into<MatrixView<'a, A>>,
    b: impl Into<MatrixView<'a, B>>,
    out: impl Into<MatrixViewMut<'a, C>>,
) {
    let (a, b) = (a.into(), b.into());
    let mut out = out.into();
    assert!(
        a.num_cols() == b.num_rows() && a.num_rows() == out.num_rows() && b.num_cols() == b.num_cols(),
        "can't multiply ({}) and ({}) into ({})",
        a.shape(),
        b.shape(),
        out.shape(),
    );

    let mut row_vals = vec![0.0; out.num_cols()];
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
        for a_col in 0..a.num_cols() {
            let a_val = a.get(row_idx, a_col);
            for b_col in 0..b.num_cols() {
                // also out-col
                row_vals[b_col] += a_val * b.get(a_col, b_col);
            }
        }
        out.set_row(row_idx, &row_vals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::tensor::Tensor;

    mod matrix {
        use super::*;

        #[test]
        fn check_matmul_more_rows_than_cols() {
            let a: Tensor<2> = [
                // comment, so that rustfmt doesn't collapse this to a single line
                [1., 2.],
                [3., 4.],
                [5., 6.],
            ]
            .into();
            let b: Tensor<2> = [
                //
                [7., 8.],
                [9., 10.],
            ]
            .into();
            let mut out: Tensor<2> = [[0.0; 2]; 3].into();
            matmul(&a, &b, &mut out);
            assert_eq!(
                out,
                [
                    [(1. * 7.) + (2. * 9.), (1. * 8.) + (2. * 10.)],
                    [(3. * 7.) + (4. * 9.), (3. * 8.) + (4. * 10.)],
                    [(5. * 7.) + (6. * 9.), (5. * 8.) + (6. * 10.)],
                ]
                .into()
            );
        }

        #[test]
        fn check_matmul_more_cols_than_rows() {
            let a: Tensor<2> = [
                // comment, so that rustfmt doesn't collapse this to a single line
                [1., 2., 3.],
                [4., 5., 6.],
            ]
            .into();
            let b: Tensor<2> = [
                //
                [7., 8.],
                [9., 10.],
                [11., 12.],
            ]
            .into();
            let mut out: Tensor<2> = Tensor::new_matrix(2, 2);
            matmul(&a, &b, &mut out);
            assert_eq!(
                out,
                [
                    [(1. * 7.) + (2. * 9.) + (3. * 11.), (1. * 8.) + (2. * 10.) + (3. * 12.),],
                    [(4. * 7.) + (5. * 9.) + (6. * 11.), (4. * 8.) + (5. * 10.) + (6. * 12.),],
                ]
                .into()
            );
        }

        #[test]
        #[should_panic = "can't multiply (2x3) and (2x3) into (2x3)"]
        fn matmul_a_b_mismatch() {
            let a = array_matrix::<2, 3>();
            let b = array_matrix::<2, 3>();
            let mut out = array_matrix::<2, 3>();
            matmul(&a, &b, &mut out);
        }

        #[test]
        #[should_panic = "can't multiply (2x3) and (3x4) into (4x2)"]
        fn matmul_output_mismatch() {
            let a = array_matrix::<2, 3>();
            let b = array_matrix::<3, 4>();
            let mut out = array_matrix::<4, 2>();
            matmul(&a, &b, &mut out);
        }
    }

    mod tensor_3 {
        use super::*;

        #[test]
        fn check_matmul_3x2() {
            let mut a: Tensor<3> = Tensor::new([2, 3, 4]);
            // Batch 0
            a.set_row([0, 0, 0], &[1., 2., 3., 4.]);
            a.set_row([0, 1, 0], &[5., 6., 7., 8.]);
            a.set_row([0, 2, 0], &[9., 10., 11., 12.]);
            // Batch 1
            a.set_row([1, 0, 0], &[13., 14., 15., 16.]);
            a.set_row([1, 1, 0], &[17., 18., 19., 20.]);
            a.set_row([1, 2, 0], &[21., 22., 23., 24.]);

            let mut b: Tensor<2> = Tensor::new_matrix(4, 5);
            b.set_row([0, 0], &[100., 200., 300., 400., 500.]);
            b.set_row([1, 0], &[600., 700., 800., 900., 1000.]);
            b.set_row([2, 0], &[1100., 1200., 1300., 1400., 1500.]);
            b.set_row([3, 0], &[1600., 1700., 1800., 1900., 2000.]);

            let mut out: Tensor<3> = Tensor::new([2, 3, 5]);

            matmul_batched(&a, &b, &mut out);

            let mut expected: Tensor<3> = Tensor::new([2, 3, 5]);
            // Batch 0, row 0: [1., 2., 3., 4.] × b
            expected.set_row(
                [0, 0, 0],
                &[
                    1. * 100. + 2. * 600. + 3. * 1100. + 4. * 1600.,
                    1. * 200. + 2. * 700. + 3. * 1200. + 4. * 1700.,
                    1. * 300. + 2. * 800. + 3. * 1300. + 4. * 1800.,
                    1. * 400. + 2. * 900. + 3. * 1400. + 4. * 1900.,
                    1. * 500. + 2. * 1000. + 3. * 1500. + 4. * 2000.,
                ],
            );
            // Batch 0, row 1: [5., 6., 7., 8.] × b
            expected.set_row(
                [0, 1, 0],
                &[
                    5. * 100. + 6. * 600. + 7. * 1100. + 8. * 1600.,
                    5. * 200. + 6. * 700. + 7. * 1200. + 8. * 1700.,
                    5. * 300. + 6. * 800. + 7. * 1300. + 8. * 1800.,
                    5. * 400. + 6. * 900. + 7. * 1400. + 8. * 1900.,
                    5. * 500. + 6. * 1000. + 7. * 1500. + 8. * 2000.,
                ],
            );
            // Batch 0, row 2: [9., 10., 11., 12.] × b
            expected.set_row(
                [0, 2, 0],
                &[
                    9. * 100. + 10. * 600. + 11. * 1100. + 12. * 1600.,
                    9. * 200. + 10. * 700. + 11. * 1200. + 12. * 1700.,
                    9. * 300. + 10. * 800. + 11. * 1300. + 12. * 1800.,
                    9. * 400. + 10. * 900. + 11. * 1400. + 12. * 1900.,
                    9. * 500. + 10. * 1000. + 11. * 1500. + 12. * 2000.,
                ],
            );
            // Batch 1, row 0: [13., 14., 15., 16.] × b
            expected.set_row(
                [1, 0, 0],
                &[
                    13. * 100. + 14. * 600. + 15. * 1100. + 16. * 1600.,
                    13. * 200. + 14. * 700. + 15. * 1200. + 16. * 1700.,
                    13. * 300. + 14. * 800. + 15. * 1300. + 16. * 1800.,
                    13. * 400. + 14. * 900. + 15. * 1400. + 16. * 1900.,
                    13. * 500. + 14. * 1000. + 15. * 1500. + 16. * 2000.,
                ],
            );
            // Batch 1, row 1: [17., 18., 19., 20.] × b
            expected.set_row(
                [1, 1, 0],
                &[
                    17. * 100. + 18. * 600. + 19. * 1100. + 20. * 1600.,
                    17. * 200. + 18. * 700. + 19. * 1200. + 20. * 1700.,
                    17. * 300. + 18. * 800. + 19. * 1300. + 20. * 1800.,
                    17. * 400. + 18. * 900. + 19. * 1400. + 20. * 1900.,
                    17. * 500. + 18. * 1000. + 19. * 1500. + 20. * 2000.,
                ],
            );
            // Batch 1, row 2: [21., 22., 23., 24.] × b
            expected.set_row(
                [1, 2, 0],
                &[
                    21. * 100. + 22. * 600. + 23. * 1100. + 24. * 1600.,
                    21. * 200. + 22. * 700. + 23. * 1200. + 24. * 1700.,
                    21. * 300. + 22. * 800. + 23. * 1300. + 24. * 1800.,
                    21. * 400. + 22. * 900. + 23. * 1400. + 24. * 1900.,
                    21. * 500. + 22. * 1000. + 23. * 1500. + 24. * 2000.,
                ],
            );

            assert_eq!(out, expected);
        }

        #[test]
        #[should_panic = "can't multiply (3x4) and (9x6) into (3x6)"]
        fn matmul_3x2_inner_dim_mismatch() {
            let a: Tensor<3> = Tensor::new([2, 3, 4]);
            let b: Tensor<2> = Tensor::new_matrix(9, 6);
            let mut out: Tensor<3> = Tensor::new([2, 3, 6]);
            matmul_batched(&a, &b, &mut out);
        }

        #[test]
        #[should_panic = "batch dimensions don't match: can't multiply (2x3x4) and (4x5) into (9x3x5)"]
        fn matmul_3x2_output_mismatch() {
            let a: Tensor<3> = Tensor::new([2, 3, 4]);
            let b: Tensor<2> = Tensor::new_matrix(4, 5);
            let mut out: Tensor<3> = Tensor::new([9, 3, 5]);
            matmul_batched(&a, &b, &mut out);
        }

        #[test]
        fn matmul_3x2_batch_size_one() {
            // Edge case: batch size of 1 should still work
            let mut a: Tensor<3> = Tensor::new([1, 2, 3]);
            a.set_row([0, 0, 0], &[1., 2., 3.]);
            a.set_row([0, 1, 0], &[4., 5., 6.]);

            let mut b: Tensor<2> = Tensor::new_matrix(3, 4);
            b.set_row([0, 0], &[1., 2., 3., 4.]);
            b.set_row([1, 0], &[5., 6., 7., 8.]);
            b.set_row([2, 0], &[9., 10., 11., 12.]);

            let mut out: Tensor<3> = Tensor::new([1, 2, 4]);

            matmul_batched(&a, &b, &mut out);

            let mut expected: Tensor<3> = Tensor::new([1, 2, 4]);
            // Batch 0, row 0: [1., 2., 3.] × b
            expected.set_row(
                [0, 0, 0],
                &[
                    1. * 1. + 2. * 5. + 3. * 9.,
                    1. * 2. + 2. * 6. + 3. * 10.,
                    1. * 3. + 2. * 7. + 3. * 11.,
                    1. * 4. + 2. * 8. + 3. * 12.,
                ],
            );
            // Batch 0, row 1: [4., 5., 6.] × b
            expected.set_row(
                [0, 1, 0],
                &[
                    4. * 1. + 5. * 5. + 6. * 9.,
                    4. * 2. + 5. * 6. + 6. * 10.,
                    4. * 3. + 5. * 7. + 6. * 11.,
                    4. * 4. + 5. * 8. + 6. * 12.,
                ],
            );

            assert_eq!(out, expected);
        }
    }

    fn array_matrix<const R: usize, const C: usize>() -> Tensor<2> {
        Tensor::new_matrix(R, C)
    }

    impl<const R: usize, const C: usize> Into<Tensor<2>> for [[f32; C]; R] {
        fn into(self) -> Tensor<2> {
            let mut m = Tensor::new_matrix(R, C);
            for (row_idx, row_vals) in self.iter().enumerate() {
                m.set_row([row_idx, 0], row_vals);
            }
            m
        }
    }
}
