use crate::cputensor::tensor::{CpuTensor, MatrixView, MatrixViewMut};

pub fn matmul_batched<const R: usize>(a: &CpuTensor<R>, b: &CpuTensor<R>) -> CpuTensor<R> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if R == 1 {
        assert!(a_shape == b_shape, "can't multiply ({a_shape}) by ({b_shape})");
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert!(
        a_shape[..R - 2] == b_shape[..R - 2],
        "can't multiply ({a_shape}) by ({b_shape})",
    );

    assert!(
        a_shape[R - 1] == b_shape[R - 2],
        "can't multiply ({a_shape}) by ({b_shape})"
    );

    let mut out_shape = a_shape;
    out_shape[R - 1] = b_shape[R - 1];
    let mut out = CpuTensor::new(out_shape);

    for batch_indices in a_shape.iter_indices().skipping_dims_at(R - 2) {
        let a_matrix = a.matrix_slice(batch_indices);
        let b_matrix = b.matrix_slice(batch_indices);
        let out_matrix = out.matrix_slice_mut(batch_indices);
        matmul(a_matrix, b_matrix, out_matrix);
    }
    out
}

fn matmul<'a, const A: usize, const B: usize, const C: usize>(
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

    out.mut_rows(|row_idx, row_vals| {
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
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cputensor::CpuBackend;
    use crate::cputensor::tensor::CpuTensor;
    use crate::tensor::TensorBackend;

    mod matrix {
        use super::*;
        use crate::cputensor::CpuBackend;
        use crate::tensor::TensorBackend;

        #[test]
        fn check_matmul_more_rows_than_cols() {
            let a: CpuTensor<2> = [
                // comment, so that rustfmt doesn't collapse this to a single line
                [1., 2.],
                [3., 4.],
                [5., 6.],
            ]
            .into();
            let b: CpuTensor<2> = [
                //
                [7., 8.],
                [9., 10.],
            ]
            .into();
            let mut out: CpuTensor<2> = [[0.0; 2]; 3].into();
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
            let a: CpuTensor<2> = [
                // comment, so that rustfmt doesn't collapse this to a single line
                [1., 2., 3.],
                [4., 5., 6.],
            ]
            .into();
            let b: CpuTensor<2> = [
                //
                [7., 8.],
                [9., 10.],
                [11., 12.],
            ]
            .into();
            let mut out: CpuTensor<2> = CpuBackend::new_matrix(2, 2);
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
        fn check_matmul_batched_3() {
            let mut a = CpuTensor::new([2, 3, 4]);
            // Batch 0: values are 1BRC (tensor 1, batch, row, col)
            a.set_row([0, 0, 0], &[1000., 1001., 1002., 1003.]);
            a.set_row([0, 1, 0], &[1010., 1011., 1012., 1013.]);
            a.set_row([0, 2, 0], &[1020., 1021., 1022., 1023.]);
            // Batch 1
            a.set_row([1, 0, 0], &[1100., 1101., 1102., 1103.]);
            a.set_row([1, 1, 0], &[1110., 1111., 1112., 1113.]);
            a.set_row([1, 2, 0], &[1120., 1121., 1122., 1123.]);

            let mut b = CpuTensor::new([2, 4, 5]);
            // Batch 0: values are 2BRC (tensor 2, batch, row, col)
            b.set_row([0, 0, 0], &[2000., 2001., 2002., 2003., 2004.]);
            b.set_row([0, 1, 0], &[2010., 2011., 2012., 2013., 2014.]);
            b.set_row([0, 2, 0], &[2020., 2021., 2022., 2023., 2024.]);
            b.set_row([0, 3, 0], &[2030., 2031., 2032., 2033., 2034.]);
            // Batch 1
            b.set_row([1, 0, 0], &[2100., 2101., 2102., 2103., 2104.]);
            b.set_row([1, 1, 0], &[2110., 2111., 2112., 2113., 2114.]);
            b.set_row([1, 2, 0], &[2120., 2121., 2122., 2123., 2124.]);
            b.set_row([1, 3, 0], &[2130., 2131., 2132., 2133., 2134.]);

            let out = matmul_batched(&a, &b);

            let mut expected: CpuTensor<3> = CpuTensor::new([2, 3, 5]);
            // Batch 0, row 0: [1000., 1001., 1002., 1003.] × b
            expected.set_row(
                [0, 0, 0],
                &[
                    (1000. * 2000.) + (1001. * 2010.) + (1002. * 2020.) + (1003. * 2030.),
                    (1000. * 2001.) + (1001. * 2011.) + (1002. * 2021.) + (1003. * 2031.),
                    (1000. * 2002.) + (1001. * 2012.) + (1002. * 2022.) + (1003. * 2032.),
                    (1000. * 2003.) + (1001. * 2013.) + (1002. * 2023.) + (1003. * 2033.),
                    (1000. * 2004.) + (1001. * 2014.) + (1002. * 2024.) + (1003. * 2034.),
                ],
            );
            // Batch 0, row 1: [1010., 1011., 1012., 1013.] × b
            expected.set_row(
                [0, 1, 0],
                &[
                    (1010. * 2000.) + (1011. * 2010.) + (1012. * 2020.) + (1013. * 2030.),
                    (1010. * 2001.) + (1011. * 2011.) + (1012. * 2021.) + (1013. * 2031.),
                    (1010. * 2002.) + (1011. * 2012.) + (1012. * 2022.) + (1013. * 2032.),
                    (1010. * 2003.) + (1011. * 2013.) + (1012. * 2023.) + (1013. * 2033.),
                    (1010. * 2004.) + (1011. * 2014.) + (1012. * 2024.) + (1013. * 2034.),
                ],
            );
            // Batch 0, row 2: [1020., 1021., 1022., 1023.] × b
            expected.set_row(
                [0, 2, 0],
                &[
                    (1020. * 2000.) + (1021. * 2010.) + (1022. * 2020.) + (1023. * 2030.),
                    (1020. * 2001.) + (1021. * 2011.) + (1022. * 2021.) + (1023. * 2031.),
                    (1020. * 2002.) + (1021. * 2012.) + (1022. * 2022.) + (1023. * 2032.),
                    (1020. * 2003.) + (1021. * 2013.) + (1022. * 2023.) + (1023. * 2033.),
                    (1020. * 2004.) + (1021. * 2014.) + (1022. * 2024.) + (1023. * 2034.),
                ],
            );
            // Batch 1, row 0: [1100., 1101., 1102., 1103.] × b
            expected.set_row(
                [1, 0, 0],
                &[
                    (1100. * 2100.) + (1101. * 2110.) + (1102. * 2120.) + (1103. * 2130.),
                    (1100. * 2101.) + (1101. * 2111.) + (1102. * 2121.) + (1103. * 2131.),
                    (1100. * 2102.) + (1101. * 2112.) + (1102. * 2122.) + (1103. * 2132.),
                    (1100. * 2103.) + (1101. * 2113.) + (1102. * 2123.) + (1103. * 2133.),
                    (1100. * 2104.) + (1101. * 2114.) + (1102. * 2124.) + (1103. * 2134.),
                ],
            );
            // Batch 1, row 1: [1110., 1111., 1112., 1113.] × b
            expected.set_row(
                [1, 1, 0],
                &[
                    (1110. * 2100.) + (1111. * 2110.) + (1112. * 2120.) + (1113. * 2130.),
                    (1110. * 2101.) + (1111. * 2111.) + (1112. * 2121.) + (1113. * 2131.),
                    (1110. * 2102.) + (1111. * 2112.) + (1112. * 2122.) + (1113. * 2132.),
                    (1110. * 2103.) + (1111. * 2113.) + (1112. * 2123.) + (1113. * 2133.),
                    (1110. * 2104.) + (1111. * 2114.) + (1112. * 2124.) + (1113. * 2134.),
                ],
            );
            // Batch 1, row 2: [1120., 1121., 1122., 1123.] × b
            expected.set_row(
                [1, 2, 0],
                &[
                    (1120. * 2100.) + (1121. * 2110.) + (1122. * 2120.) + (1123. * 2130.),
                    (1120. * 2101.) + (1121. * 2111.) + (1122. * 2121.) + (1123. * 2131.),
                    (1120. * 2102.) + (1121. * 2112.) + (1122. * 2122.) + (1123. * 2132.),
                    (1120. * 2103.) + (1121. * 2113.) + (1122. * 2123.) + (1123. * 2133.),
                    (1120. * 2104.) + (1121. * 2114.) + (1122. * 2124.) + (1123. * 2134.),
                ],
            );

            assert_eq!(out, expected);
        }

        #[test]
        #[should_panic = "can't multiply (2x3x4) by (2x3x4)"]
        fn matmul_batched_inner_dim_mismatch() {
            let a: CpuTensor<3> = CpuTensor::new([2, 3, 4]);
            let b: CpuTensor<3> = CpuTensor::new([2, 3, 4]);
            let _ = matmul_batched(&a, &b);
        }

        #[test]
        #[should_panic = "can't multiply (2x3x4) by (9x4x5)"]
        fn matmul_batched_batch_dim_mismatch() {
            let a: CpuTensor<3> = CpuTensor::new([2, 3, 4]);
            let b: CpuTensor<3> = CpuTensor::new([9, 4, 5]);
            let _ = matmul_batched(&a, &b);
        }
    }

    fn array_matrix<const R: usize, const C: usize>() -> CpuTensor<2> {
        CpuBackend::new_matrix(R, C)
    }

    impl<const R: usize, const C: usize> Into<CpuTensor<2>> for [[f32; C]; R] {
        fn into(self) -> CpuTensor<2> {
            let mut m = CpuBackend::new_matrix(R, C);
            for (row_idx, row_vals) in self.iter().enumerate() {
                m.set_row([row_idx, 0], row_vals);
            }
            m
        }
    }
}
