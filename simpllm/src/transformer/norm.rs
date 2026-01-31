use crate::tensor::Matrix;

pub struct Norm {
    scale: f32,
    shift: f32,
    epsilon: f32,
}

impl Norm {
    pub fn new(scale: f32, shift: f32, epsilon: f32) -> Self {
        Self { scale, shift, epsilon }
    }

    pub fn apply(&self, input: &Matrix) -> Matrix {
        let mut result = input.clone();
        for row_idx in 0..result.num_rows() {
            result.mut_row([row_idx, 0], |row| {
                let Stats { mean, variance } = row.iter().copied().into();
                for val in row {
                    let normalized = (*val - mean) / (variance + self.epsilon).sqrt();
                    *val = (normalized * self.scale) + self.shift;
                }
            });
        }
        result
    }
}

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
