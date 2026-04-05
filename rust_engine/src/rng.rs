/// SplitMix64 PRNG — identical to Python's game.rng.SimpleRng.
///
/// Same seed → same sequence in both languages.
/// Period: 2^64. Passes BigCrush.

use std::f64::consts::PI;

#[derive(Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1). Top 53 bits for full double precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Random integer in [low, high).
    pub fn randint(&mut self, low: i64, high: i64) -> i64 {
        let r = (high - low) as u64;
        low + (self.next_u64() % r) as i64
    }

    /// Choose k items from 0..n without replacement (partial Fisher-Yates).
    pub fn choice_no_replace(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut arr: Vec<usize> = (0..n).collect();
        let mut result = Vec::with_capacity(k);
        for i in 0..k {
            let j = i + (self.next_u64() as usize % (n - i));
            arr.swap(i, j);
            result.push(arr[i]);
        }
        result
    }

    /// Generate `size` random integers in [low, high).
    pub fn integers(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        (0..size).map(|_| self.randint(low, high)).collect()
    }

    /// Standard normal via Box-Muller. Consumes 2 u64 values.
    pub fn next_normal(&mut self) -> f64 {
        let mut u1 = self.next_f64();
        let u2 = self.next_f64();
        while u1 == 0.0 {
            u1 = self.next_f64();
        }
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Gamma(alpha, 1) via Marsaglia-Tsang method.
    pub fn next_gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let mut u = self.next_f64();
            while u == 0.0 {
                u = self.next_f64();
            }
            return self.next_gamma(alpha + 1.0) * u.powf(1.0 / alpha);
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = self.next_normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Dirichlet sample from alpha parameters.
    pub fn dirichlet(&mut self, alphas: &[f64]) -> Vec<f64> {
        let samples: Vec<f64> = alphas.iter().map(|&a| self.next_gamma(a)).collect();
        let total: f64 = samples.iter().sum();
        if total == 0.0 {
            let n = samples.len() as f64;
            return vec![1.0 / n; samples.len()];
        }
        samples.iter().map(|&s| s / total).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let mut r1 = SimpleRng::new(42);
        let mut r2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut r1 = SimpleRng::new(0);
        let mut r2 = SimpleRng::new(1);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn test_choice_no_replace_unique() {
        let mut rng = SimpleRng::new(42);
        let chosen = rng.choice_no_replace(81, 3);
        assert_eq!(chosen.len(), 3);
        assert_ne!(chosen[0], chosen[1]);
        assert_ne!(chosen[0], chosen[2]);
        assert_ne!(chosen[1], chosen[2]);
        for &c in &chosen {
            assert!(c < 81);
        }
    }

    #[test]
    fn test_randint_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let v = rng.randint(1, 8);
            assert!(v >= 1 && v < 8);
        }
    }

    #[test]
    fn test_dirichlet_sums_to_one() {
        let mut rng = SimpleRng::new(42);
        let d = rng.dirichlet(&[0.3; 10]);
        let sum: f64 = d.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &v in &d {
            assert!(v >= 0.0);
        }
    }
}
