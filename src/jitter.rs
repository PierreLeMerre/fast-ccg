use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use crate::xcorr::{FftPlans, xcorr_fft_planned, nextpow2, make_theta, make_target};

/// Jitter a 2D spike matrix [time x trials]
pub fn jitter_2d(spikes: &Array2<f64>, window: usize) -> Array2<f64> {
    let n_time   = spikes.nrows();
    let n_trials = spikes.ncols();

    let psth: Array1<f64> = spikes.mean_axis(Axis(1)).expect("mean failed");
    let n_windows = n_time / window;
    let mut output = Array2::<f64>::zeros((n_time, n_trials));

    for w in 0..n_windows {
        let t_start = w * window;
        let t_end   = t_start + window;

        let window_sum: Array1<f64> = spikes
            .slice(s![t_start..t_end, ..])
            .sum_axis(Axis(0));

        let psth_sum: f64 = psth.slice(s![t_start..t_end]).sum();
        let psth_sum_safe = if psth_sum == 0.0 { 1e-10 } else { psth_sum };

        for t in t_start..t_end {
            let scale = psth[t] / psth_sum_safe;
            for trial in 0..n_trials {
                output[[t, trial]] = window_sum[trial] * scale;
            }
        }
    }
    output
}

/// Compute mean cross-correlation across trials using pre-computed FFT plans
/// and pre-allocated scratch buffers — zero heap allocation in the inner loop
pub fn mean_xcorr_fast(
    spikes_i: &Array2<f64>,
    spikes_m: &Array2<f64>,
    plans:    &FftPlans,
    target:   &[usize],
) -> Vec<f64> {
    let n_time   = spikes_i.nrows();
    let n_trials = spikes_i.ncols();
    let nfft     = plans.nfft;
    let scale    = 1.0 / nfft as f64;
    let half     = nfft / 2;

    // Pre-allocate ALL buffers once — reused every trial
    let mut col_i:  Vec<f64>           = vec![0.0; n_time];
    let mut col_m:  Vec<f64>           = vec![0.0; n_time];
    let mut buf_a:  Vec<Complex<f64>>  = vec![Complex::default(); nfft];
    let mut buf_b:  Vec<Complex<f64>>  = vec![Complex::default(); nfft];
    let mut sum:    Vec<f64>           = vec![0.0; target.len()];

    for trial in 0..n_trials {
        // Fill column buffers in-place
        for t in 0..n_time {
            col_i[t] = spikes_i[[t, trial]];
            col_m[t] = spikes_m[[t, trial]];
        }

        // Run FFT cross-correlation into buf_a (in-place, no allocation)
        xcorr_fft_planned(&col_i, &col_m, plans, &mut buf_a, &mut buf_b);

        // Extract target bins with fftshift applied on-the-fly
        // fftshift: index i maps to (i + half) % nfft in the shifted output
        for (k, &t) in target.iter().enumerate() {
            // t is the shifted index — map back to unshifted
            let unshifted = (t + half) % nfft;
            sum[k] += buf_a[unshifted].re * scale;
        }
    }

    sum.iter().map(|&s| s / n_trials as f64).collect()
}

/// Compatibility wrapper — used by tests
pub fn mean_xcorr(
    spikes_i: &Array2<f64>,
    spikes_m: &Array2<f64>,
    nfft:     usize,
    target:   &[usize],
    _n_trials: usize,
) -> Vec<f64> {
    let plans = FftPlans::new(nfft);
    mean_xcorr_fast(spikes_i, spikes_m, &plans, target)
}

/// One neuron pair — returns [raw | corrected] concatenated
pub fn ccg_pair(
    spikes_i:      &Array2<f64>,
    spikes_m:      &Array2<f64>,
    fr_i:          f64,
    fr_m:          f64,
    jitter_window: usize,
) -> Vec<f64> {
    let n_time = spikes_i.nrows();
    let nfft   = nextpow2(2 * n_time);
    let target   = make_target(n_time, nfft);
    let theta    = make_theta(n_time);

    // Create FFT plans ONCE per pair — shared across raw + jitter xcorr
    let plans = FftPlans::new(nfft);

    let raw_ccg    = mean_xcorr_fast(spikes_i, spikes_m, &plans, &target);

    let jittered_i = jitter_2d(spikes_i, jitter_window);
    let jittered_m = jitter_2d(spikes_m, jitter_window);
    let jitter_ccg = mean_xcorr_fast(&jittered_i, &jittered_m, &plans, &target);

    // Geometric mean of per-bin firing rates — gives corrected CCG in units of
    // excess coincidences per bin normalised by sqrt(FR_i * FR_m).
    // NOTE: raw_ccg already carries a 1/nfft scale from mean_xcorr_fast, so
    // corrected values will be of order p/nfft for a connection with
    // transmission probability p — small in absolute terms but the z-score
    // is computed within the corrected CCG's own scale, so that is fine.
    let geom_mean = (fr_i * fr_m).sqrt();

    let corrected: Vec<f64> = raw_ccg.iter()
        .zip(jitter_ccg.iter())
        .zip(theta.iter())
        .map(|((raw, jit), th)| {
            let denom = geom_mean * th;
            if denom.abs() < 1e-10 { 0.0 } else { (raw - jit) / denom }
        })
        .collect();

    let mut out = raw_ccg;
    out.extend(corrected);
    out
}

/// All pairs in parallel
pub fn compute_all_pairs(
    spikes:        &[Array2<f64>],
    firing_rates:  &[f64],
    jitter_window: usize,
) -> Vec<(usize, usize, Vec<f64>)> {
    let n_units = spikes.len();
    let pairs: Vec<(usize, usize)> = (0..n_units)
        .flat_map(|i| (i+1..n_units).map(move |m| (i, m)))
        .collect();

    pairs.par_iter()
        .map(|&(i, m)| {
            let result = ccg_pair(
                &spikes[i], &spikes[m],
                firing_rates[i], firing_rates[m],
                jitter_window,
            );
            (i, m, result)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::xcorr::{nextpow2, make_target};

    #[test]
    fn test_mean_xcorr_self() {
        let n_time = 100; let n_trials = 10;
        let nfft   = nextpow2(2 * n_time);
        let target = make_target(n_time, nfft);
        let mut spikes = Array2::<f64>::zeros((n_time, n_trials));
        for t in (0..n_time).step_by(10) {
            for trial in 0..n_trials { spikes[[t, trial]] = 1.0; }
        }
        let raw = mean_xcorr(&spikes, &spikes, nfft, &target, n_trials);
        let mid = raw.len() / 2;
        let max_idx = raw.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();
        println!("Raw CCG length: {}  Peak at {} (expected {})", raw.len(), max_idx, mid);
        assert_eq!(max_idx, mid, "Raw CCG peak should be at lag 0");
    }

    #[test]
    fn test_jitter_preserves_rate() {
        let n_time = 100; let n_trials = 10;
        let mut spikes = Array2::<f64>::zeros((n_time, n_trials));
        for t in (0..n_time).step_by(10) {
            for trial in 0..n_trials { spikes[[t, trial]] = 1.0; }
        }
        let jittered = jitter_2d(&spikes, 25);
        let ratio = jittered.sum() / spikes.sum();
        assert!((ratio - 1.0).abs() < 0.1, "ratio={:.4}", ratio);
    }

    #[test]
    fn test_ccg_offset_trains() {
        let n_time = 100; let n_trials = 10;
        let nfft   = nextpow2(2 * n_time);
        let target = make_target(n_time, nfft);
        let mut spikes_a = Array2::<f64>::zeros((n_time, n_trials));
        let mut spikes_b = Array2::<f64>::zeros((n_time, n_trials));
        for t in (0..n_time).step_by(20) {
            for trial in 0..n_trials { spikes_a[[t, trial]] = 1.0; }
        }
        for t in (5..n_time).step_by(20) {
            for trial in 0..n_trials { spikes_b[[t, trial]] = 1.0; }
        }
        let raw = mean_xcorr(&spikes_a, &spikes_b, nfft, &target, n_trials);
        let mid = raw.len() / 2;
        let max_idx = raw.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();
        let peak_lag = max_idx as i64 - mid as i64;
        assert_eq!(peak_lag.abs(), 5, "Peak should be at lag ±5");
    }

    #[test]
    fn test_parallel_pairs() {
        let n_neurons = 5; let n_time = 50; let n_trials = 5;
        let spikes: Vec<Array2<f64>> = (0..n_neurons)
            .map(|_| Array2::<f64>::zeros((n_time, n_trials))).collect();
        let frs = vec![5.0f64; n_neurons];
        let results = compute_all_pairs(&spikes, &frs, 10);
        assert_eq!(results.len(), n_neurons * (n_neurons - 1) / 2);
        println!("Computed {} pairs ✓", results.len());
    }
}