#[cfg(test)]
#[macro_export]
macro_rules! assert_f32_slice {
        ($actual:expr, $expected:expr) => {{
            use approx::abs_diff_eq;
            assert_f32_slice!($actual, $expected, abs_diff_eq);
        }};
        ($actual:expr, $expected:expr, $method:ident $(, $($optname:tt = $optval:literal),*)?) => {
            assert_eq!(
                $actual.len(),
                $expected.len(),
                "slices have different lengths: {:?} != {:?}",
                $actual,
                $expected
            );

            for (idx, (a, e)) in $actual.iter().zip($expected.iter()).enumerate() {
                if !$method!(a, e $(, $($optname = $optval),* )?) {
                    assert_eq!($actual, $expected, "at index [{idx}]: {a} != {e}");
                    panic!("expected {:?} but saw {:?}", $expected, $actual); // just in case
                }
            }
        };
    }
