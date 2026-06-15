# ── Usage ─────────────────────────────────────────────────────────────────────
# julia --sysimage /path/to/ccg_sysimage.so run_ccg_job.jl <nwb_path> [out_dir]
#
# Arguments:
#   nwb_path  — path to the .nwb file to process
#   out_dir   — directory for the output .h5 file (default: same dir as nwb_path)

length(ARGS) >= 1 || error("Usage: julia run_ccg_job.jl <nwb_path> [out_dir]")

const NWB_PATH = ARGS[1]
const OUT_DIR  = length(ARGS) >= 2 ? ARGS[2] : dirname(NWB_PATH)

isfile(NWB_PATH) || error("NWB file not found: $NWB_PATH")
isdir(OUT_DIR)   || mkpath(OUT_DIR)

# ── Imports ───────────────────────────────────────────────────────────────────
include("../lib/ccg.jl")
include("../lib/load_data.jl")
include("../lib/significance.jl")
using .CCG
using Statistics
using Printf
using HDF5

# ── Configuration ─────────────────────────────────────────────────────────────
const INTERVALS     = nothing   # nothing = whole recording
const BINSIZE       = 0.0005    # 0.5 ms bins
const MIN_FR_HZ     = 1.0       # minimum firing rate in Hz
const JITTER_WIN    = 50        # 25 ms jitter window → 50 bins at 0.5 ms
const CHUNK_DUR     = 3.0       # seconds per chunk (whole-recording mode)
const N_INTERVALS   = 200       # randomly sample this many chunks; nothing = all
const UNITS         = nothing   # nothing = all units

# ── Load data ─────────────────────────────────────────────────────────────────
println("[$NWB_PATH] Loading spike times...")
spk_times, unit_ids = load_spike_times(NWB_PATH)
println("  $(length(spk_times)) units loaded")

intervals = isnothing(INTERVALS) ?
    whole_recording_intervals(spk_times; chunk_duration=CHUNK_DUR, max_intervals=N_INTERVALS) : INTERVALS
n_trials_raw  = size(intervals, 1)
epoch_dur_ms  = round((intervals[1, 2] - intervals[1, 1]) * 1000, digits=1)
win_bins      = round(Int, (intervals[1, 2] - intervals[1, 1]) / BINSIZE)

println("  Intervals: $n_trials_raw × $epoch_dur_ms ms → $win_bins bins/trial")

# ── Bin spikes ────────────────────────────────────────────────────────────────
println("Binning spikes...")
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
println("Computing CCGs (Rust, parallel)...")
t_start = time()
raw, jitter, corrected, pairs, lags = CCG.compute_all_pairs(
    matrices, frs; jitter_window=JITTER_WIN
)
t_elapsed = time() - t_start
lags_ms = lags .* (BINSIZE * 1000)

println("  Done in $(round(t_elapsed, digits=2))s  ($(round(n_pairs/t_elapsed, digits=1)) pairs/s)")

# ── Total spike counts per neuron ─────────────────────────────────────────────
n_spikes     = [sum(matrices[i]) for i in 1:n_neurons]
pairs_tuples = [(p[1], p[2]) for p in pairs]

# ── Significance testing ──────────────────────────────────────────────────────
println("Testing significance...")
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
println("  Excitatory: $(sum(is_sig_ex)) / $n_pairs  |  Inhibitory: $(sum(is_sig_in)) / $n_pairs")

# ── Save results ──────────────────────────────────────────────────────────────
sig_idx = findall(is_sig_any)
n_sig   = length(sig_idx)

if n_sig == 0
    println("No significant pairs — nothing saved.")
else
    out_name = joinpath(OUT_DIR, "ccg_" * splitext(basename(NWB_PATH))[1] * ".h5")
    ccg_len  = size(raw, 1)
    @printf "Saving %d significant pairs  [%d lags × %d pairs]  (%.1f MB each)\n" n_sig ccg_len n_sig ccg_len*n_sig*8/1e6

    h5open(out_name, "w") do f
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
    println("Saved → $out_name")
end

println("Done.")
