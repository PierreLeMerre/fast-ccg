# ── Imports ───────────────────────────────────────────────────────────────────
include("../lib/ccg.jl")
include("../lib/load_data.jl")
include("../lib/significance.jl")
using .CCG
using Statistics
using Printf
using HDF5

# ── Configuration ─────────────────────────────────────────────────────────────
const NWB_PATH  = "/Volumes/MorseSSD/NPX_Database/frozen_Aversion_Nat_Submission/216300_20200518-probe0.nwb"

# Intervals: Matrix{Float64} [n_trials × 2], columns [t_in  t_out] in seconds.
# Set to nothing to use the whole recording (0 → last spike).
# Example: const INTERVALS = [0.0 10.0; 20.0 30.0; 50.0 60.0]
const INTERVALS     = nothing
const BINSIZE       = 0.0005  # 0.5ms bins
const MIN_FR_HZ     = 0.75     # minimum firing rate in Hz
const JITTER_WIN    = 50      # 25ms jitter window → 50 bins at 0.5ms
const CHUNK_DUR     = 3.0     # seconds per chunk when using whole-recording mode
const N_INTERVALS   = 200     # randomly sample this many chunks; nothing = all
const UNITS         = nothing  # restrict to specific unit IDs, e.g. [1, 44, 67]; nothing = all
const N_THREADS     = 0        # Rayon worker threads: 0 = all available cores

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading spike times...")
spk_times, unit_ids = load_spike_times(NWB_PATH)
println("  $(length(spk_times)) units loaded")

intervals = isnothing(INTERVALS) ?
    whole_recording_intervals(spk_times; chunk_duration=CHUNK_DUR, max_intervals=N_INTERVALS) : INTERVALS
n_trials_raw  = size(intervals, 1)
epoch_dur_ms  = round((intervals[1, 2] - intervals[1, 1]) * 1000, digits=1)
win_bins      = round(Int, (intervals[1, 2] - intervals[1, 1]) / BINSIZE)

println("Intervals : $n_trials_raw trial(s) × $epoch_dur_ms ms → $win_bins bins/trial")
println("  Spike matrix memory estimate per neuron: $(round(win_bins * n_trials_raw * 8 / 1e6, digits=1)) MB")

# ── Bin spikes ────────────────────────────────────────────────────────────────
println("\nBinning spikes...")
matrices, frs, kept_ids = prepare_all_neurons(
    spk_times, unit_ids, intervals;
    binsize      = BINSIZE,
    min_fr_hz    = MIN_FR_HZ,
    select_units = UNITS
)

n_time, n_trials = size(matrices[1])
n_neurons        = length(matrices)
n_pairs          = n_neurons * (n_neurons - 1) ÷ 2

println("  Neurons kept    : $n_neurons / $(length(spk_times))")
println("  Matrix size     : [$n_time bins × $n_trials trials]")
println("  Pairs to compute: $n_pairs")

# ── Run CCG (Rust, parallel) ──────────────────────────────────────────────────
println("\nComputing CCGs (Rust, parallel)...")
t_start = time()
raw, jitter, corrected, pairs, lags = CCG.compute_all_pairs(
    matrices, frs; jitter_window=JITTER_WIN, n_threads=N_THREADS
)
t_elapsed = time() - t_start

lags_ms = lags .* (BINSIZE * 1000)

println("  Done in $(round(t_elapsed, digits=2))s")
println("  Throughput : $(round(n_pairs/t_elapsed, digits=1)) pairs/s")
println("  Output     : raw=$(size(raw)), jitter=$(size(jitter)), corrected=$(size(corrected))")

# ── Total spike counts per neuron (for transmission probability) ──────────────
n_spikes = [sum(matrices[i]) for i in 1:n_neurons]

# pairs are (i,m) tuples with 1-based indices into n_spikes
pairs_tuples = [(p[1], p[2]) for p in pairs]

# ── Significance testing ──────────────────────────────────────────────────────
println("\nTesting significance...")
is_sig_ex, is_sig_in, peak_lags, peak_zs, trough_lags, trough_zs, tp = test_significance(
    raw, jitter, corrected, lags_ms,
    Int.(round.(n_spikes)), n_trials, pairs_tuples;
    peak_window_ms   = 10.0,
    baseline_lo_ms   = 50.0,
    baseline_hi_ms   = 100.0,
    threshold_sd     = 7.0,
    neg_threshold_sd = 3.0,
    zero_lag_ms      = 0.8
)

is_sig_any = is_sig_ex .| is_sig_in

# ── Summary ───────────────────────────────────────────────────────────────────
println("\nTop 5 excitatory pairs by Z-score:")
println("  Pair              Peak lag (ms)    Z-score    TP")
println("  " * "─"^55)
for rank in 1:min(5, n_pairs)
    k      = sortperm(peak_zs, rev=true)[rank]
    i, m   = pairs[k]
    @printf "  (unit %3d, unit %3d)   %+6.1f ms    %7.2f    %.4f\n" kept_ids[i] kept_ids[m] peak_lags[k] peak_zs[k] tp[k]
end

println("\nTop 5 inhibitory pairs by Z-score:")
println("  Pair             Trough lag (ms)   Z-score")
println("  " * "─"^45)
for rank in 1:min(5, n_pairs)
    k      = sortperm(trough_zs)[rank]   # most negative first
    i, m   = pairs[k]
    @printf "  (unit %3d, unit %3d)   %+6.1f ms    %7.2f\n" kept_ids[i] kept_ids[m] trough_lags[k] trough_zs[k]
end

println("\n  Excitatory: $(sum(is_sig_ex)) / $n_pairs  |  Inhibitory: $(sum(is_sig_in)) / $n_pairs")

# ── Save significant pairs only ───────────────────────────────────────────────
println("\nSaving results...")

sig_idx = findall(is_sig_any)
n_sig   = length(sig_idx)

if n_sig == 0
    println("  No significant pairs to save.")
else
    out_name = "ccg_" * splitext(basename(NWB_PATH))[1] * ".h5"
    ccg_len  = size(raw, 1)
    @printf "  Saving %d significant pairs  [%d lags × %d pairs]  (%.1f MB each)\n" n_sig ccg_len n_sig ccg_len*n_sig*8/1e6

    h5open(out_name, "w") do f
        # CCG matrices — significant pairs only, chunked + compressed
        chunk = (ccg_len, 1)
        for (name, data) in [("raw", raw[:, sig_idx]),
                              ("jitter", jitter[:, sig_idx]),
                              ("corrected", corrected[:, sig_idx])]
            ds = create_dataset(f, name, datatype(Float64),
                                dataspace(ccg_len, n_sig);
                                chunk=chunk, deflate=3)
            write(ds, data)
        end

        f["lags_ms"]       = lags_ms
        f["kept_ids"]      = Int.(kept_ids)
        f["unit_i"]        = Int.([kept_ids[pairs[k][1]] for k in sig_idx])
        f["unit_m"]        = Int.([kept_ids[pairs[k][2]] for k in sig_idx])
        f["is_excitatory"] = Float64.(is_sig_ex[sig_idx])
        f["is_inhibitory"] = Float64.(is_sig_in[sig_idx])
        f["peak_lags"]     = peak_lags[sig_idx]
        f["peak_zs"]       = peak_zs[sig_idx]
        f["trough_lags"]   = trough_lags[sig_idx]
        f["trough_zs"]     = trough_zs[sig_idx]
        f["tp"]            = tp[sig_idx]
        f["binsize_ms"]    = BINSIZE * 1000
        f["win_sz_ms"]     = epoch_dur_ms
        f["jitter_win"]    = Float64(JITTER_WIN)
        f["n_neurons"]     = Float64(n_neurons)
        f["n_trials"]      = Float64(n_trials)
    end
    println("  Saved → $out_name")
end

println("\nDone.")
