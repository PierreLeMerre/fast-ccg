# ── Imports ───────────────────────────────────────────────────────────────────
include("../lib/ccg.jl")
include("../lib/load_data.jl")
include("../lib/significance.jl")
using .CCG
using Statistics
using Printf
using HDF5
using Glob
using ProgressMeter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to switch datasets
# ══════════════════════════════════════════════════════════════════════════════
const CONFIG = (
    # Data sources
    nwb_src    = "/Volumes/MorseSSD/NPX_Database/frozen_PFCMap_Nat_Submission",
    dest       = "/Volumes/MorseSSD/CCG",

    # Intervals: Matrix{Float64} [n_trials × 2] with columns [t_in  t_out] in seconds.
    # Set to nothing to use the whole recording (0 → last spike) for every file.
    # Example: intervals = [0.0 10.0; 20.0 30.0]
    intervals  = nothing,

    # CCG parameters
    binsize    = 0.0005,   # 0.5ms
    jitter_win = 50,       # 25ms → 50 bins at 0.5ms
    min_fr_hz  = 1.0,      # minimum firing rate in Hz
    chunk_dur     = 3.0,   # seconds per chunk in whole-recording mode
    n_intervals   = 200,   # randomly sample this many chunks; nothing = all
    units      = nothing,  # restrict to specific unit IDs, e.g. [1, 44, 67]; nothing = all
    n_threads  = 0,        # Rayon worker threads: 0 = all available cores

    # Significance thresholds
    peak_window_ms   = 10.0,
    baseline_lo_ms   = 50.0,
    baseline_hi_ms   = 100.0,
    threshold_sd     = 7.0,
    neg_threshold_sd = 3.0,
)
# ══════════════════════════════════════════════════════════════════════════════

# ── Helper: build output path from filename ───────────────────────────────────
function output_path(cfg, filename)
    joinpath(cfg.dest, "CCG_" * filename * ".h5")
end

# ── Helper: process one recording ────────────────────────────────────────────
function process_recording(cfg, nwb_file)
    filename = splitext(basename(nwb_file))[1]
    out_path = output_path(cfg, filename)

    println("\n" * "="^60)
    println("Recording : $filename")
    println("="^60)

    # ── Load spike times ──────────────────────────────────────────────────
    println("  Loading spike times...")
    spk_times, unit_ids = load_spike_times(nwb_file)
    println("    $(length(spk_times)) units in NWB")

    # ── Resolve intervals ─────────────────────────────────────────────────
    intervals    = isnothing(cfg.intervals) ?
        whole_recording_intervals(spk_times; chunk_duration=cfg.chunk_dur, max_intervals=cfg.n_intervals) : cfg.intervals
    n_trials     = size(intervals, 1)
    epoch_dur_ms = round((intervals[1, 2] - intervals[1, 1]) * 1000, digits=1)
    win_bins     = round(Int, (intervals[1, 2] - intervals[1, 1]) / cfg.binsize)
    println("    $n_trials trial(s)  |  $epoch_dur_ms ms  |  $win_bins bins/trial")

    # ── Bin spikes ────────────────────────────────────────────────────────
    println("  Binning spikes...")
    matrices, frs, kept_ids = prepare_all_neurons(
        spk_times, unit_ids, intervals;
        binsize      = cfg.binsize,
        min_fr_hz    = cfg.min_fr_hz,
        select_units = cfg.units
    )

    n_neurons = length(matrices)
    n_pairs   = n_neurons * (n_neurons - 1) ÷ 2

    if n_neurons < 2
        println("  ⚠ Fewer than 2 neurons passed FR filter — skipping")
        return :skipped_no_neurons
    end

    n_time, n_trials_mat = size(matrices[1])
    println("    $n_neurons neurons kept  |  $n_pairs pairs  |  [$n_time × $n_trials_mat]")

    # ── Compute CCG ───────────────────────────────────────────────────────
    println("  Computing CCGs (Rust, parallel)...")
    t_start = time()
    raw, jitter, corrected, pairs, lags = CCG.compute_all_pairs(
        matrices, frs; jitter_window=cfg.jitter_win, n_threads=cfg.n_threads
    )
    t_elapsed = time() - t_start
    lags_ms   = lags .* (cfg.binsize * 1000)

    println(@sprintf "    Done in %.2fs  (%.0f pairs/s)" t_elapsed n_pairs/t_elapsed)

    # ── Significance testing ──────────────────────────────────────────────
    println("  Testing significance...")
    n_spikes     = [sum(matrices[i]) for i in 1:n_neurons]
    pairs_tuples = [(p[1], p[2]) for p in pairs]
    is_sig_ex, is_sig_in, peak_lags, peak_zs, trough_lags, trough_zs, tp = test_significance(
        raw, jitter, corrected, lags_ms,
        Int.(round.(n_spikes)), n_trials_mat, pairs_tuples;
        peak_window_ms   = cfg.peak_window_ms,
        baseline_lo_ms   = cfg.baseline_lo_ms,
        baseline_hi_ms   = cfg.baseline_hi_ms,
        threshold_sd     = cfg.threshold_sd,
        neg_threshold_sd = cfg.neg_threshold_sd,
        zero_lag_ms      = 0.8
    )
    is_sig_any = is_sig_ex .| is_sig_in
    n_sig      = sum(is_sig_any)
    println(@sprintf "    %d excitatory  |  %d inhibitory  |  %d / %d total" sum(is_sig_ex) sum(is_sig_in) n_sig n_pairs)

    # ── Save — only significant pairs ─────────────────────────────────────
    println("  Saving → $out_path")
    sig_idx    = findall(is_sig_any)
    unit_i_sig = Int.([kept_ids[pairs[k][1]] for k in sig_idx])
    unit_m_sig = Int.([kept_ids[pairs[k][2]] for k in sig_idx])

    mkpath(dirname(out_path))
    h5open(out_path, "w") do f
        if !isempty(sig_idx)
            ccg_len = size(raw, 1)
            chunk   = (ccg_len, 1)
            for (name, data) in [("raw", raw[:, sig_idx]),
                                  ("jitter", jitter[:, sig_idx]),
                                  ("corrected", corrected[:, sig_idx])]
                ds = create_dataset(f, name, datatype(Float64),
                    dataspace(ccg_len, n_sig), chunk=chunk, deflate=3)
                write(ds, data)
            end
        end

        f["lags_ms"]        = lags_ms
        f["kept_ids"]       = Int.(kept_ids)
        f["unit_i"]         = unit_i_sig
        f["unit_m"]         = unit_m_sig
        f["is_excitatory"]  = Float64.(is_sig_ex[sig_idx])
        f["is_inhibitory"]  = Float64.(is_sig_in[sig_idx])
        f["peak_lags"]      = peak_lags[sig_idx]
        f["peak_zs"]        = peak_zs[sig_idx]
        f["trough_lags"]    = trough_lags[sig_idx]
        f["trough_zs"]      = trough_zs[sig_idx]
        f["tp"]             = tp[sig_idx]

        f["filename"]       = filename
        f["n_neurons"]      = Float64(n_neurons)
        f["n_pairs"]        = Float64(n_pairs)
        f["n_sig"]          = Float64(n_sig)
        f["n_trials"]       = Float64(n_trials_mat)
        f["n_time"]         = Float64(n_time)
        f["epoch_dur_ms"]   = epoch_dur_ms
        f["binsize_ms"]     = cfg.binsize * 1000
        f["jitter_win"]     = Float64(cfg.jitter_win)
        f["t_elapsed"]      = t_elapsed
    end

    println(@sprintf "    Saved %d significant pairs" n_sig)
    return :success
end

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

# Find all NWB files recursively
nwb_files = glob("**/*.nwb", CONFIG.nwb_src)
if isempty(nwb_files)
    # Try non-recursive if no results
    nwb_files = glob("*.nwb", CONFIG.nwb_src)
end

println("="^60)
println("CCG BATCH PROCESSING")
println("="^60)
println("NWB source : $(CONFIG.nwb_src)")
println("Destination: $(CONFIG.dest)")
println("Found      : $(length(nwb_files)) NWB files")
println("="^60)

# ── Run loop ──────────────────────────────────────────────────────────────────
t_batch_start = time()
stats = Dict(:success => 0, :skipped_no_neurons => 0, :error => 0)

prog = Progress(length(nwb_files); desc="Recordings: ", showspeed=true)
for nwb_file in nwb_files
    status = try
        process_recording(CONFIG, nwb_file)
    catch e
        println("  ✗ ERROR: $e")
        :error
    end
    stats[status] = get(stats, status, 0) + 1
    next!(prog; showvalues = [(:done, stats[:success]), (:errors, stats[:error])])
end
finish!(prog)

# ── Batch summary ─────────────────────────────────────────────────────────────
t_batch = time() - t_batch_start
println("\n" * "="^60)
println("BATCH COMPLETE")
println("="^60)
@printf "  Total time     : %.1f s  (%.1f min)\n" t_batch t_batch/60
@printf "  Successful     : %d\n"  stats[:success]
@printf "  No neurons     : %d\n"  stats[:skipped_no_neurons]
@printf "  Errors         : %d\n"  stats[:error]
println("  Results saved → $(CONFIG.dest)")