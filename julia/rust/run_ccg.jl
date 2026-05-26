# ── Imports ───────────────────────────────────────────────────────────────────
include("../lib/ccg.jl")
include("../lib/load_data.jl")
include("../lib/significance.jl")
using .CCG
using Statistics
using Printf
using HDF5

# ── Configuration ─────────────────────────────────────────────────────────────
const NWB_PATH   = "/Volumes/MorseSSD/NPX_Database/frozen_Aversion_Nat_Submission/216300_20200518-probe0.nwb"
const EPOCH_PATH = "/Volumes/MorseSSD/Pierre_Server_BACKUP/PFCmap_NHP/preprocessing/metrics_extraction/timeselections/timeselections_Carlen/216300_20200518-probe0__TSELprestim3__STATEactive__all.h5"
const BINSIZE    = 0.0005   # 0.5ms bins
const MIN_FR     = 0.0005   # minimum firing rate (spikes/bin)
const JITTER_WIN = 50       # 25ms jitter window → 50 bins at 0.5ms

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading spike times...")
spk_times, unit_ids = load_spike_times(NWB_PATH)
println("  $(length(spk_times)) units loaded")

println("Loading epochs...")
ts_in, ts_out = load_epochs(EPOCH_PATH)
n_trials_raw  = length(ts_in)
epoch_dur_ms  = round((ts_out[1] - ts_in[1]) * 1000, digits=1)
win_bins      = round(Int, (ts_out[1] - ts_in[1]) / BINSIZE)

println("  $n_trials_raw epochs loaded")
println("  Epoch duration : $epoch_dur_ms ms → $win_bins bins/trial")

# ── Bin spikes ────────────────────────────────────────────────────────────────
println("\nBinning spikes...")
matrices, frs, kept_ids = prepare_all_neurons(
    spk_times, ts_in, ts_out;
    binsize = BINSIZE,
    min_fr  = MIN_FR
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
raw, corrected, pairs, lags = CCG.compute_all_pairs(
    matrices, frs; jitter_window=JITTER_WIN
)
t_elapsed = time() - t_start

lags_ms = lags .* (BINSIZE * 1000)

println("  Done in $(round(t_elapsed, digits=2))s")
println("  Throughput : $(round(n_pairs/t_elapsed, digits=1)) pairs/s")
println("  Output     : raw=$(size(raw)), corrected=$(size(corrected))")

# ── Significance testing ──────────────────────────────────────────────────────
println("\nTesting significance...")
is_sig, peak_lags_sig, peak_zs = test_significance(
    corrected, lags_ms;
    peak_window_ms = 10.0,
    baseline_lo_ms = 50.0,
    baseline_hi_ms = 100.0,
    threshold_sd   = 7.0,
    binsize_ms     = BINSIZE * 1000
)

# ── Summary ───────────────────────────────────────────────────────────────────
println("\nTop 5 pairs by Z-score:")
println("  Pair              Peak lag (ms)    Z-score    Significant?")
println("  " * "─"^55)

order = sortperm(peak_zs, rev=true)
for rank in 1:min(5, n_pairs)
    k      = order[rank]
    i, m   = pairs[k]
    unit_i = kept_ids[i]
    unit_m = kept_ids[m]
    sig    = is_sig[k] ? "✓" : " "
    @printf "  (unit %3d, unit %3d)   %+6.1f ms    %7.2f    %s\n" unit_i unit_m peak_lags_sig[k] peak_zs[k] sig
end

println("\n  Total significant: $(sum(is_sig)) / $n_pairs pairs ($(round(100*sum(is_sig)/n_pairs, digits=1))%)")

# ── Save ──────────────────────────────────────────────────────────────────────
println("\nSaving results...")
h5open("ccg_results.h5", "w") do f
    f["raw"]        = raw
    f["corrected"]  = corrected
    f["lags_ms"]    = lags_ms
    f["kept_ids"]   = kept_ids
    f["pairs_i"]    = [p[1] for p in pairs]
    f["pairs_m"]    = [p[2] for p in pairs]
    f["is_sig"]     = Float64.(is_sig)
    f["peak_lags"]  = peak_lags_sig
    f["peak_zs"]    = peak_zs
    f["binsize_ms"] = BINSIZE * 1000
    f["win_sz_ms"]  = epoch_dur_ms
    f["jitter_win"] = Float64(JITTER_WIN)
    f["n_neurons"]  = Float64(n_neurons)
    f["n_trials"]   = Float64(n_trials)
end
println("  Saved → ccg_results.h5")
println("\nDone. $(sum(is_sig)) significant pairs out of $n_pairs total.")