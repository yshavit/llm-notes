pub fn softmax(vals: &mut [f32]) {
    for v in vals.iter_mut() {
        *v = v.exp();
    }
    let exps_sums: f32 = vals.iter().sum();
    for v in vals.iter_mut() {
        *v = *v / exps_sums;
    }
}

#[cfg(test)]
mod tests {
    mod softmax {
        use crate::assert_f32_slice;
        use crate::tensor::vector::softmax;
        use approx::abs_diff_eq;

        #[test]
        fn happy_path() {
            // taken from wikipedia :-)
            let mut values = [1., 2., 3., 4., 1., 2., 3.];
            let expected = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175];

            softmax(&mut values);

            assert_f32_slice!(values, expected, abs_diff_eq, epsilon = 0.001);
        }

        #[test]
        fn negative_inf() {
            // Negative infinity doesn't affect other values. It becomes 0.
            let mut values = [1., 2., 3., 4., 1., 2., 3., f32::NEG_INFINITY, f32::NEG_INFINITY];
            let expected = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175, 0., 0.];

            softmax(&mut values);

            assert_f32_slice!(values, expected, abs_diff_eq, epsilon = 0.001);
        }
    }
}
