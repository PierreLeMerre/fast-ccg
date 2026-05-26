use std::sync::Arc;
use rustfft::{FftPlanner, Fft, num_complex::Complex};

/// Next power of 2 greater than n
pub fn nextpow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

/// Pre-computed FFT plans — created ONCE per (nfft, n_time) combination
/// and reused across all trials and pairs
pub struct FftPlans {
    pub forward: Arc<dyn Fft<f64>>,
    pub inverse: Arc<dyn Fft<f64>>,
    pub nfft:    usize,
}

impl FftPlans {
    pub fn new(nfft: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        Self {
            forward: planner.plan_fft_forward(nfft),
            inverse: planner.plan_fft_inverse(nfft),
            nfft,
        }
    }
}

/// FFT cross correlation using pre-computed plans
/// Reuses pre-allocated buffers to avoid heap allocation in the hot loop
pub fn xcorr_fft_planned(
    a:       &[f64],
    b:       &[f64],
    plans:   &FftPlans,
    buf_a:   &mut Vec<Complex<f64>>,   // pre-allocated scratch buffer
    buf_b:   &mut Vec<Complex<f64>>,   // pre-allocated scratch buffer
) -> () {
    // Fill buffers in-place — no allocation
    let nfft = plans.nfft;
    buf_a.clear();
    buf_b.clear();

    for i in 0..nfft {
        buf_a.push(Complex { re: if i < a.len() { a[i] } else { 0.0 }, im: 0.0 });
        buf_b.push(Complex { re: if i < b.len() { b[i] } else { 0.0 }, im: 0.0 });
    }

    plans.forward.process(buf_a);
    plans.forward.process(buf_b);

    // Pointwise multiply A * conj(B) in-place into buf_a
    for (fa, fb) in buf_a.iter_mut().zip(buf_b.iter()) {
        *fa = *fa * fb.conj();
    }

    plans.inverse.process(buf_a);  // result is in buf_a
}

/// Original xcorr_fft — kept for compatibility with tests
pub fn xcorr_fft(a: &[f64], b: &[f64], nfft: usize) -> Vec<f64> {
    let plans  = FftPlans::new(nfft);
    let mut ba = vec![Complex { re: 0.0, im: 0.0 }; nfft];
    let mut bb = vec![Complex { re: 0.0, im: 0.0 }; nfft];

    xcorr_fft_planned(a, b, &plans, &mut ba, &mut bb);

    let scale = 1.0 / nfft as f64;
    let half  = nfft / 2;
    ba[half..].iter()
        .chain(ba[..half].iter())
        .map(|c| c.re * scale)
        .collect()
}

/// Triangular normalization — length 2*n_t - 1
pub fn make_theta(n_t: usize) -> Vec<f64> {
    let start = -((n_t as i64) - 1);
    let end   = n_t as i64;
    (start..=end - 1)
        .map(|t| n_t as f64 - t.abs() as f64)
        .collect()
}

/// Target indices into fftshift output
pub fn make_target(n_t: usize, nfft: usize) -> Vec<usize> {
    let half  = nfft / 2;
    let start = -((n_t as i64) - 1);
    let end   = n_t as i64;
    (start..end)
        .map(|i| (half as i64 + i) as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xcorr_fft_self_correlation() {
        let signal = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let n_t    = signal.len();
        let nfft   = nextpow2(2 * n_t);
        let result = xcorr_fft(&signal, &signal, nfft);
        let target = make_target(n_t, nfft);
        let windowed: Vec<f64> = target.iter().map(|&i| result[i]).collect();

        println!("Window length: {}", windowed.len());
        for (i, v) in windowed.iter().enumerate() {
            let lag = i as i64 - (n_t as i64 - 1);
            println!("lag {:+3}: {:.4}", lag, v);
        }

        let max_idx = windowed.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();

        assert_eq!(max_idx, n_t - 1,
            "Peak should be at lag 0 (index {}), got {}", n_t - 1, max_idx);
    }
}