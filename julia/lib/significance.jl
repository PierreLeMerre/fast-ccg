using Statistics

"""
    test_significance(corrected, lags_ms; kwargs...)

Test CCG significance for both excitatory (positive) and inhibitory (negative) peaks.

Excitatory: peak > threshold_sd (default 7 SD) above baseline within ±peak_window_ms
Inhibitory: trough < neg_threshold_sd (default 3 SD) below baseline within ±peak_window_ms

Baseline: 50 < |τ| < 100 ms. Zero-lag bin excluded.

Returns:
  is_sig_ex   : BitVector — excitatory significant pairs
  is_sig_in   : BitVector — inhibitory significant pairs
  peak_lags   : Vector{Float64} — lag of excitatory peak (ms)
  peak_zs     : Vector{Float64} — z-score of excitatory peak
  trough_lags : Vector{Float64} — lag of inhibitory trough (ms)
  trough_zs   : Vector{Float64} — z-score of inhibitory trough (negative)
"""
function test_significance(
    corrected::Matrix{Float64},
    lags_ms::Vector{Float64};
    peak_window_ms::Float64   = 10.0,
    baseline_lo_ms::Float64   = 50.0,
    baseline_hi_ms::Float64   = 100.0,
    threshold_sd::Float64     = 7.0,
    neg_threshold_sd::Float64 = 3.0,
    binsize_ms::Float64       = 0.5
)
    n_pairs = size(corrected, 2)

    baseline_mask = (abs.(lags_ms) .> baseline_lo_ms) .&
                    (abs.(lags_ms) .< baseline_hi_ms)

    zero_lag_ms = binsize_ms / 2
    peak_mask   = (abs.(lags_ms) .< peak_window_ms) .&
                  (abs.(lags_ms) .> zero_lag_ms)

    is_sig_ex   = falses(n_pairs)
    is_sig_in   = falses(n_pairs)
    peak_lags   = zeros(Float64, n_pairs)
    peak_zs     = zeros(Float64, n_pairs)
    trough_lags = zeros(Float64, n_pairs)
    trough_zs   = zeros(Float64, n_pairs)

    for k in 1:n_pairs
        ccg           = corrected[:, k]
        baseline_vals = ccg[baseline_mask]
        baseline_mean = mean(baseline_vals)
        baseline_sd   = std(baseline_vals)

        baseline_sd < 1e-10 && continue

        peak_region = ccg[peak_mask]
        peak_indices = findall(peak_mask)

        # Excitatory: positive peak
        peak_val  = maximum(peak_region)
        peak_idx  = peak_indices[argmax(peak_region)]
        z_ex      = (peak_val - baseline_mean) / baseline_sd
        peak_lags[k] = lags_ms[peak_idx]
        peak_zs[k]   = z_ex
        is_sig_ex[k] = z_ex > threshold_sd

        # Inhibitory: negative trough
        trough_val  = minimum(peak_region)
        trough_idx  = peak_indices[argmin(peak_region)]
        z_in        = (trough_val - baseline_mean) / baseline_sd
        trough_lags[k] = lags_ms[trough_idx]
        trough_zs[k]   = z_in
        is_sig_in[k]   = z_in < -neg_threshold_sd
    end

    n_ex = sum(is_sig_ex)
    n_in = sum(is_sig_in)
    println("Excitatory pairs : $n_ex / $n_pairs  ($(round(100*n_ex/n_pairs, digits=1))%)")
    println("Inhibitory pairs : $n_in / $n_pairs  ($(round(100*n_in/n_pairs, digits=1))%)")
    return is_sig_ex, is_sig_in, peak_lags, peak_zs, trough_lags, trough_zs
end
