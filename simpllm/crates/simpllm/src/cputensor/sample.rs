use crate::cputensor::{Matrix, softmax};
use std::cmp::Ordering;

pub struct LogitSampler {
    logits: Matrix,
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
}

impl LogitSampler {
    pub fn new(logits: Matrix) -> Self {
        Self {
            logits,
            top_k: None,
            top_p: None,
            temperature: None,
        }
    }

    pub fn top_k(mut self, k: Option<usize>) -> Self {
        self.top_k = k;
        self
    }

    pub fn top_prob(mut self, p: Option<f32>) -> Self {
        self.top_p = p;
        self
    }

    pub fn temperature(mut self, t: Option<f32>) -> Self {
        self.temperature = t;
        self
    }

    pub fn get(mut self) -> usize {
        self.logits.mut_row([self.logits.num_rows() - 1, 0], |row| {
            if let Some(use_top_k) = self.top_k {
                top_k(row, use_top_k);
            }
            if let Some(use_top_p) = self.top_p {
                top_prob(row, use_top_p);
            }
            if let Some(temp) = self.temperature {
                row.iter_mut().for_each(|v| *v /= temp);
            }
            multinomial_sample(row)
        })
    }
}

fn multinomial_sample(mut logits: &mut [f32]) -> usize {
    softmax(&mut logits);

    let target_cumulative: f32 = rand::random();

    let mut cumulative = 0.0;
    let last_idx = logits.len() - 1;
    for (idx, value) in logits.into_iter().enumerate() {
        cumulative += *value;
        if target_cumulative < cumulative {
            return idx;
        }
    }
    last_idx
}

fn top_k(logits: &mut [f32], k: usize) {
    let logits_and_index = sorted_desc(logits);
    logits.fill(f32::NEG_INFINITY);
    logits_and_index
        .into_iter()
        .take(k)
        .for_each(|(idx, val)| logits[idx] = val);
}

fn top_prob(logits: &mut [f32], prob: f32) {
    let logits_total: f32 = logits.iter().sum();
    let logits_and_index = sorted_desc(logits);
    logits.fill(f32::NEG_INFINITY);
    let mut cumulative_prob = 0.0;
    for (idx, logit) in logits_and_index {
        cumulative_prob += logit / logits_total;
        if cumulative_prob > prob {
            break;
        }
        logits[idx] = logit;
    }
}

fn sorted_desc(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut logits_and_index: Vec<_> = logits.iter().copied().enumerate().collect();
    // sort by value desc, then by index asc as tiebreaker (that's only really needed for unit testing)
    logits_and_index.sort_by(|a, b| match a.1.total_cmp(&b.1).reverse() {
        Ordering::Equal => a.0.cmp(&b.0),
        not_equal => not_equal,
    });
    logits_and_index
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_multinomial_sampled_index_distribution() {
        // Get logits that add up to 100, so they're just percents. That's not needed for correctness,
        // but it makes the test more obvious.
        let logits = vec![15.0, 40.0, 10.0, 25.0, 10.0];

        const NUM_SAMPLES: usize = 100_000;
        const TOLERANCE_PERCENT: f32 = 0.03;
        let tolerance: usize = (TOLERANCE_PERCENT * NUM_SAMPLES as f32) as usize;

        // Check distribution is approximately correct
        let expected_counts: Vec<_> = {
            let total: f32 = logits.iter().sum();
            logits
                .iter()
                .map(|logit| {
                    let expected_count = (logit / total) * (NUM_SAMPLES as f32);
                    expected_count as i32
                })
                .collect()
        };

        // Sample many times
        let mut actual_counts = vec![0; logits.len()];
        for _ in 0..NUM_SAMPLES {
            // softmax applies exp() to each element; so, we'll pre-apply fn() to each, such that softmax gets us back
            // to the original values
            let mut normalized_logits: Vec<f32> = logits.iter().map(|logit| logit.ln()).collect();
            let idx = multinomial_sample(&mut normalized_logits);
            actual_counts[idx] += 1;
        }

        for i in 0..logits.len() {
            let actual = actual_counts[i];
            let expected = expected_counts[i];
            let diff = (actual - expected).abs() as usize;
            if diff > tolerance {
                // print all of both, for easier debugging
                assert_eq!(actual_counts, expected_counts, "logit samples");
                assert!(false, "test failed"); // just to be sure!
            }
        }
    }

    #[test]
    fn test_top_k() {
        let n_inf = f32::NEG_INFINITY;
        let mut logits = vec![15.0, 40.0, 10.0, 25.0, 10.0];
        top_k(&mut logits, 3);
        assert_eq!(logits, vec![15.0, 40.0, n_inf, 25.0, n_inf]);
    }

    #[test]
    fn test_top_k_with_tiebreaker() {
        let n_inf = f32::NEG_INFINITY;
        let mut logits = vec![15.0, 40.0, 10.0, 25.0, 10.0];
        // only the first 10.0 gets picked
        top_k(&mut logits, 4);
        assert_eq!(logits, vec![15.0, 40.0, 10.0, 25.0, n_inf]);
    }

    #[test]
    fn test_top_prob() {
        let n_inf = f32::NEG_INFINITY;
        let mut logits = vec![15.0, 40.0, 10.0, 25.0, 10.0];
        top_prob(&mut logits, 0.66);
        assert_eq!(logits, vec![n_inf, 40.0, n_inf, 25.0, n_inf]);
    }

    /// Test fencepost errors around [`top_prob`]. These don't actually matter in practice, but we may as well!
    #[test]
    fn test_top_prob_fencepost() {
        let n_inf = f32::NEG_INFINITY;
        {
            // 25 just barely makes the cut
            let mut logits = vec![15.0, 40.0, 10.0, 25.0, 10.0];
            top_prob(&mut logits, 0.65);
            assert_eq!(logits, vec![n_inf, 40.0, n_inf, 25.0, n_inf]);
        }
        {
            // 25 just barely doesn't make the cut
            let mut logits = vec![15.0, 40.0, 10.0, 25.0, 10.0];
            top_prob(&mut logits, 0.649);
            assert_eq!(logits, vec![n_inf, 40.0, n_inf, n_inf, n_inf]);
        }
    }
}
