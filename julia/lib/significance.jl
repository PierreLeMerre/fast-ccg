using Statistics

"""
    consecutive_threshold(vals, threshold; excitatory=true)

Return true if ≥2 adjacent bins in `vals` cross `threshold` in the given direction.
"""
function consecutive_threshold(vals::AbstractVector{Float64}, threshold::Float64;
                                excitatory::Bool = true)
    cross = excitatory ? (vals .> threshold) : (vals .< threshold)
    for j in 1:length(cross)-1
        cross[j] && cross[j+1] && return true
    end
    return false
end

"""
    test_significance(raw, jitter, corrected, lags_ms, n_spikes, n_trials, pairs; kwargs...)

Test CCG significance with:
  - 0.8 ms lag floor (avoids spike-sorting artefacts at zero-lag)
  - ≥2 consecutive bins above/below threshold
  - Transmission probability (excitatory pairs only)

Excitatory: peak > threshold_sd above baseline in (0.8, 10] ms AND ≥2 consecutive bins
Inhibitory: trough < neg_threshold_sd below baseline in (0.8, 10] ms AND ≥2 consecutive bins
TP: sum_{tau in [0.8,10]ms} (raw[tau]-jitter[tau]) / n_spikes_i   (for i->m direction)

Arguments:
  raw, jitter, corrected : Matrices [ccg_len × n_pairs], mean coincidences/trial
  lags_ms    : Vector of lag values in ms
  n_spikes   : total spike count per neuron across all trials (length = n_neurons)
  n_trials   : number of trials
  pairs      : Vector of (i,m) neuron-index tuples (1-based, into n_spikes)

Returns:
  is_sig_ex, is_sig_in : BitVectors
  peak_lags, peak_zs   : lag (ms) and z-score of excitatory peak
  trough_lags, trough_zs : lag (ms) and z-score of inhibitory trough
  tp                   : transmission probability (positive lags, i→m)
"""
function test_significance(
    raw::Matrix{Float64},
    jitter::Matrix{Float64},
    corrected::Matrix{Float64},
    lags_ms::Vector{Float64},
    n_spikes::Vector{Int},
    n_trials::Int,
    pairs::Vector{Tuple{Int,Int}};
    peak_window_ms::Float64   = 10.0,
    baseline_lo_ms::Float64   = 50.0,
    baseline_hi_ms::Float64   = 100.0,
    threshold_sd::Float64     = 7.0,
    neg_threshold_sd::Float64 = 3.0,
    zero_lag_ms::Float64      = 0.8,   # 0.8 ms floor
)
    n_pairs = size(corrected, 2)

    baseline_mask = (abs.(lags_ms) .> baseline_lo_ms) .&
                    (abs.(lags_ms) .< baseline_hi_ms)

    # Positive-lag window for excitatory detection [0.8, 10] ms (i leads m)
    excit_mask  = (lags_ms .>  zero_lag_ms) .& (lags_ms .<= peak_window_ms)

    # Full ±window for inhibitory (symmetric; exclude zero-lag floor)
    inhib_mask  = (abs.(lags_ms) .>  zero_lag_ms) .& (abs.(lags_ms) .<= peak_window_ms)

    # TP window — same as excit_mask
    tp_mask = excit_mask

    is_sig_ex   = falses(n_pairs)
    is_sig_in   = falses(n_pairs)
    peak_lags   = zeros(Float64, n_pairs)
    peak_zs     = zeros(Float64, n_pairs)
    trough_lags = zeros(Float64, n_pairs)
    trough_zs   = zeros(Float64, n_pairs)
    tp          = zeros(Float64, n_pairs)

    for k in 1:n_pairs
        ccg           = corrected[:, k]
        baseline_vals = ccg[baseline_mask]
        baseline_mean = mean(baseline_vals)
        baseline_sd   = std(baseline_vals)

        baseline_sd < 1e-10 && continue

        ex_threshold = baseline_mean + threshold_sd     * baseline_sd
        in_threshold = baseline_mean - neg_threshold_sd * baseline_sd

        # ── Excitatory ──────────────────────────────────────────────────────
        excit_region  = ccg[excit_mask]
        excit_indices = findall(excit_mask)

        if !isempty(excit_region)
            peak_val = maximum(excit_region)
            peak_idx = excit_indices[argmax(excit_region)]
            z_ex     = (peak_val - baseline_mean) / baseline_sd
            peak_lags[k] = lags_ms[peak_idx]
            peak_zs[k]   = z_ex
            is_sig_ex[k] = z_ex > threshold_sd &&
                           consecutive_threshold(excit_region, ex_threshold; excitatory=true)
        end

        # ── Inhibitory ──────────────────────────────────────────────────────
        inhib_region  = ccg[inhib_mask]
        inhib_indices = findall(inhib_mask)

        if !isempty(inhib_region)
            trough_val = minimum(inhib_region)
            trough_idx = inhib_indices[argmin(inhib_region)]
            z_in       = (trough_val - baseline_mean) / baseline_sd
            trough_lags[k] = lags_ms[trough_idx]
            trough_zs[k]   = z_in
            is_sig_in[k]   = z_in < -neg_threshold_sd &&
                             consecutive_threshold(inhib_region, in_threshold; excitatory=false)
        end

        # ── Transmission probability ─────────────────────────────────────────
        # TP = Σ_{τ ∈ [0.8,10]ms} (raw[τ]−jitter[τ]) × n_trials / n_spikes_i
        # raw/jitter are mean coincidences/trial → ×n_trials converts to total excess Y-spikes
        # dividing by n_spikes_i (total X spikes) gives fraction: "10% of X spikes are followed by Y"
        i_idx = pairs[k][1]
        ns_i  = n_spikes[i_idx]
        if ns_i > 0
            excess = (raw[tp_mask, k] .- jitter[tp_mask, k])
            tp[k]  = sum(excess) * n_trials / ns_i
        end
    end

    n_ex = sum(is_sig_ex)
    n_in = sum(is_sig_in)
    println("Excitatory pairs : $n_ex / $n_pairs  ($(round(100*n_ex/n_pairs, digits=1))%)")
    println("Inhibitory pairs : $n_in / $n_pairs  ($(round(100*n_in/n_pairs, digits=1))%)")
    return is_sig_ex, is_sig_in, peak_lags, peak_zs, trough_lags, trough_zs, tp
end
