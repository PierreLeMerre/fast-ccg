using Statistics

"""
    test_significance(corrected, lags_ms; kwargs...)

Test CCG significance following Siegle et al. criteria:
  1. Peak within ±peak_window_ms of zero lag
  2. Peak > threshold_sd SDs above baseline (50 < |τ| < 100ms)
  3. Zero-lag bin excluded

Returns: is_sig, peak_lags, peak_zs
"""
function test_significance(
    corrected::Matrix{Float64},
    lags_ms::Vector{Float64};
    peak_window_ms::Float64 = 10.0,
    baseline_lo_ms::Float64 = 50.0,
    baseline_hi_ms::Float64 = 100.0,
    threshold_sd::Float64   = 7.0,
    binsize_ms::Float64     = 0.5
)
    n_pairs = size(corrected, 2)

    baseline_mask = (abs.(lags_ms) .> baseline_lo_ms) .&
                    (abs.(lags_ms) .< baseline_hi_ms)

    zero_lag_ms = binsize_ms / 2
    peak_mask   = (abs.(lags_ms) .< peak_window_ms) .&
                  (abs.(lags_ms) .> zero_lag_ms)

    is_sig    = falses(n_pairs)
    peak_lags = zeros(Float64, n_pairs)
    peak_zs   = zeros(Float64, n_pairs)

    for k in 1:n_pairs
        ccg           = corrected[:, k]
        baseline_vals = ccg[baseline_mask]
        baseline_mean = mean(baseline_vals)
        baseline_sd   = std(baseline_vals)

        baseline_sd < 1e-10 && continue

        peak_region = ccg[peak_mask]
        peak_val    = maximum(peak_region)
        peak_idx    = findall(peak_mask)[argmax(peak_region)]
        peak_lag    = lags_ms[peak_idx]
        z           = (peak_val - baseline_mean) / baseline_sd

        peak_lags[k] = peak_lag
        peak_zs[k]   = z
        is_sig[k]    = z > threshold_sd
    end

    n_sig = sum(is_sig)
    println("Significant pairs: $n_sig / $n_pairs  ($(round(100*n_sig/n_pairs, digits=1))%)")
    return is_sig, peak_lags, peak_zs
end