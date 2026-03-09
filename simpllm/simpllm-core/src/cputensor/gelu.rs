pub(super) fn gelu(x: f32) -> f32 {
    use std::f32::consts::PI;
    /// MYSTMD::GELU START
    0.5 * x * (1. + f32::tanh((2. / PI).sqrt() * (x + 0.044715 * x.powi(3))))
    /// MYSTMD::GELU END
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gelu_values() {
        assert_abs_diff_eq!(gelu(-2.0), -0.0455, epsilon = 0.0001);
        assert_abs_diff_eq!(gelu(-1.0), -0.1587, epsilon = 0.001);
        assert_abs_diff_eq!(gelu(-0.75), -0.1700, epsilon = 0.001);
        assert_abs_diff_eq!(gelu(0.), 0.0, epsilon = 0.001);
        assert_abs_diff_eq!(gelu(0.), 0.0, epsilon = 0.001);
        assert_abs_diff_eq!(gelu(1.), 0.8413, epsilon = 0.001);
        assert_abs_diff_eq!(gelu(2.), 1.9545, epsilon = 0.001);
    }
}
