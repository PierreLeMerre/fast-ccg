pub mod xcorr;
pub mod jitter;

use ndarray::Array2;
use jitter::compute_all_pairs;

/// C-compatible entry point for Julia — single pair
#[no_mangle]
pub extern "C" fn ccg_pair_ffi(
    spikes_i_ptr: *const f64,
    n_time_i: usize,
    n_trials_i: usize,
    spikes_m_ptr: *const f64,
    n_time_m: usize,
    n_trials_m: usize,
    fr_i: f64,
    fr_m: f64,
    jitter_window: usize,
    out_ptr: *mut f64,
    out_len: usize,  // should be 2 * (2*n_time - 1)
) {
    let spikes_i_slice = unsafe {
        std::slice::from_raw_parts(spikes_i_ptr, n_time_i * n_trials_i)
    };
    let spikes_m_slice = unsafe {
        std::slice::from_raw_parts(spikes_m_ptr, n_time_m * n_trials_m)
    };

    let spikes_i = Array2::from_shape_vec((n_trials_i, n_time_i), spikes_i_slice.to_vec())
        .expect("reshape failed")
        .reversed_axes();
    let spikes_m = Array2::from_shape_vec((n_trials_m, n_time_m), spikes_m_slice.to_vec())
        .expect("reshape failed")
        .reversed_axes();

    // result is now [raw | corrected], length = 2 * ccg_len
    let result = jitter::ccg_pair(&spikes_i, &spikes_m, fr_i, fr_m, jitter_window);

    let out_slice = unsafe {
        std::slice::from_raw_parts_mut(out_ptr, out_len)
    };
    let copy_len = result.len().min(out_len);
    out_slice[..copy_len].copy_from_slice(&result[..copy_len]);
}

/// C-compatible entry point for Julia — all pairs in parallel
#[no_mangle]
pub extern "C" fn compute_all_pairs_ffi(
    spikes_ptr: *const f64,
    n_time: usize,
    n_trials: usize,
    n_neurons: usize,
    fr_ptr: *const f64,
    jitter_window: usize,
    out_ptr: *mut f64,
    out_len: usize,  // should be 2 * ccg_len * n_pairs
) {
    let spikes_flat = unsafe {
        std::slice::from_raw_parts(spikes_ptr, n_time * n_trials * n_neurons)
    };
    let fr_slice = unsafe {
        std::slice::from_raw_parts(fr_ptr, n_neurons)
    };

    let spikes_per_neuron: Vec<Array2<f64>> = (0..n_neurons)
        .map(|n| {
            let offset = n * n_time * n_trials;
            let chunk  = &spikes_flat[offset..offset + n_time * n_trials];
            Array2::from_shape_vec((n_trials, n_time), chunk.to_vec())
                .expect("reshape failed")
                .reversed_axes()
        })
        .collect();

    let firing_rates: Vec<f64> = fr_slice.to_vec();
    let results = compute_all_pairs(&spikes_per_neuron, &firing_rates, jitter_window);

    let out_slice = unsafe {
        std::slice::from_raw_parts_mut(out_ptr, out_len)
    };

    // Each result is now [raw | corrected] so length is 2 * ccg_len
    for (k, (_i, _m, ccg)) in results.iter().enumerate() {
        let ccg_len = ccg.len();  // already doubled inside ccg_pair
        let offset  = k * ccg_len;
        if offset + ccg_len <= out_len {
            out_slice[offset..offset + ccg_len].copy_from_slice(ccg);
        }
    }
}

/// Diagnostic function — call from Julia to verify FFI is working
/// Returns the sum of the input array into out[0], and the max value into out[1]
#[no_mangle]
pub extern "C" fn diagnose_ffi(
    ptr: *const f64,
    len: usize,
    out: *mut f64,
) {
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let sum: f64 = slice.iter().sum();
    let max: f64 = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, 2) };
    out_slice[0] = sum;
    out_slice[1] = max;
}

/// Detailed diagnostic — checks reshape and mean_xcorr output
#[no_mangle]
pub extern "C" fn diagnose_ccg_ffi(
    spikes_i_ptr: *const f64,
    n_time: usize,
    n_trials: usize,
    out: *mut f64,  // out[0]=sum after reshape, out[1]=ccg max, out[2]=ccg sum
) {
    let slice = unsafe {
        std::slice::from_raw_parts(spikes_i_ptr, n_time * n_trials)
    };

    // Step 1: check reshape
    let arr = Array2::from_shape_vec((n_trials, n_time), slice.to_vec())
        .expect("reshape failed")
        .reversed_axes();

    let sum_after_reshape: f64 = arr.iter().sum();

    // Step 2: check mean_xcorr with self
    use crate::xcorr::{nextpow2, make_target};
    let nfft   = nextpow2(2 * n_time);
    let target = make_target(n_time, nfft);
    let raw    = jitter::mean_xcorr(&arr, &arr, nfft, &target, n_trials);

    let ccg_max: f64 = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ccg_sum: f64 = raw.iter().sum();

    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, 3) };
    out_slice[0] = sum_after_reshape;
    out_slice[1] = ccg_max;
    out_slice[2] = ccg_sum;
}

/// Diagnostic 3 — check what ccg_pair actually returns
#[no_mangle]
pub extern "C" fn diagnose_ccg_pair_ffi(
    spikes_i_ptr: *const f64,
    n_time_i: usize,
    n_trials_i: usize,
    spikes_m_ptr: *const f64,
    n_time_m: usize,
    n_trials_m: usize,
    fr_i: f64,
    fr_m: f64,
    jitter_window: usize,
    out: *mut f64,  // out[0]=result.len(), out[1]=max, out[2]=sum
) {
    let spikes_i_slice = unsafe {
        std::slice::from_raw_parts(spikes_i_ptr, n_time_i * n_trials_i)
    };
    let spikes_m_slice = unsafe {
        std::slice::from_raw_parts(spikes_m_ptr, n_time_m * n_trials_m)
    };

    let spikes_i = Array2::from_shape_vec((n_trials_i, n_time_i), spikes_i_slice.to_vec())
        .expect("reshape failed")
        .reversed_axes();
    let spikes_m = Array2::from_shape_vec((n_trials_m, n_time_m), spikes_m_slice.to_vec())
        .expect("reshape failed")
        .reversed_axes();

    let result = jitter::ccg_pair(&spikes_i, &spikes_m, fr_i, fr_m, jitter_window);

    let result_max: f64 = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let result_sum: f64 = result.iter().sum();

    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, 3) };
    out_slice[0] = result.len() as f64;  // actual length of result
    out_slice[1] = result_max;
    out_slice[2] = result_sum;
}