use crate::cputensor::{CpuBackend, CpuVector};
use crate::tensor::{Matrix, Tensor2D, Vector};

pub struct CpuLayerNorm {
    pub scale: CpuVector,
    pub bias: CpuVector,
    pub epsilon: f32,
}

impl crate::tensor::LayerNorm for CpuLayerNorm {
    type B = CpuBackend;

    fn new(scale: Vector<Self::B>, bias: Vector<Self::B>, epsilon: f32) -> Self {
        Self { scale, bias, epsilon }
    }

    fn apply(&self, input: &Matrix<Self::B>) -> Matrix<Self::B> {
        let mut result = input.clone();
        result.mut_rows(|_, row| {
            let Stats { mean, variance } = row.iter().copied().into();
            for col_idx in 0..row.len() {
                let normalized = (row[col_idx] - mean) / (variance + self.epsilon).sqrt();
                row[col_idx] = (normalized * self.scale.get([col_idx])) + self.bias.get([col_idx]);
            }
        });
        result
    }
}

#[derive(Debug, Copy, Clone)]
struct Stats {
    mean: f32,
    variance: f32,
}

impl<I: Iterator<Item = f32>> From<I> for Stats {
    /// An implementation of [Welford's online algorithm].
    ///
    /// [Welford's online algorithm]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    fn from(vals: I) -> Self {
        #[derive(Default)]
        struct Acc {
            mean: f32,
            count: usize,
            m2: f32,
        }
        let acc = vals.fold(Acc::default(), |mut acc, v| {
            acc.count += 1;
            let old_mean = acc.mean;
            acc.mean += (v - acc.mean) / (acc.count as f32);
            acc.m2 += (v - old_mean) * (v - acc.mean);
            acc
        });
        Stats {
            mean: acc.mean,
            variance: acc.m2 / (acc.count as f32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn check_stats() {
        let stats: Stats = (1..=6).map(|x| x as f32).into();
        assert_abs_diff_eq!(stats.mean, 3.5);
        assert_abs_diff_eq!(stats.variance, 2.9166667);
    }
}
