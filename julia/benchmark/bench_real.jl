include("../lib/load_data.jl")
include("../lib/ccg.jl")
using .CCG
using PyCall
using Statistics
using Printf
using HDF5

pushfirst!(PyVector(pyimport("sys")."path"),
           joinpath(@__DIR__, "..", "python"))
const ccg_py = pyimport("ccg_jittercorrected")

const NWB_PATH   = "/Volumes/MorseSSD/NPX_Database/frozen_Aversion_Nat_Submission/216300_20200518-probe0.nwb"
const EPOCH_PATH = "/Volumes/MorseSSD/Pierre_Server_BACKUP/PFCmap_NHP/preprocessing/metrics_extraction/timeselections/timeselections_Carlen/216300_20200518-probe0__TSELprestim3__STATEactive__all.h5"
const BINSIZE    = 0.0005
const MIN_FR     = 0.0005
const JITTER_WIN = 50

println("="^60)
println("REAL DATA BENCHMARK: Python vs Rust CCG")
println("="^60)

# ── Load data ─────────────────────────────────────────────────────────────────
println("\nLoading data...")
spk_times, unit_ids = load_spike_times(NWB_PATH)
ts_in, ts_out       = load_epochs(EPOCH_PATH)

const WIN_SZ     = 0.5      # ±500ms CCG window
const WIN_BINS   = round(Int, WIN_SZ / BINSIZE)  # 1000 bins

println("Binning spikes...")
matrices_full, frs, kept_ids = prepare_all_neurons(
    spk_times, ts_in, ts_out;
    binsize = BINSIZE,
    min_fr  = MIN_FR
)

# Crop to ±100ms CCG window — same as run_ccg.jl
matrices = [m[1:WIN_BINS, :] for m in matrices_full]

n_time, n_trials = size(matrices[1])
println("  Cropped to ±$(WIN_SZ*1000)ms window ($WIN_BINS bins/trial)")

n_neurons        = length(matrices)
n_pairs          = n_neurons * (n_neurons - 1) ÷ 2

println("  $n_neurons neurons  |  $n_pairs pairs  |  $n_trials trials  |  $n_time bins/trial")

# ── Python ────────────────────────────────────────────────────────────────────
println("\nRunning Python pipeline...")
spikemat = zeros(Float64, n_neurons, n_trials, n_time)  # n_time is now WIN_BINS
for (i, mat) in enumerate(matrices)
    spikemat[i, :, :] = permutedims(mat, (2, 1))
end
t_py = @elapsed begin
    ccg_py_out = ccg_py.get_ccgjitter(spikemat, JITTER_WIN)
end
println(@sprintf "  Python : %.2f s  (%.1f pairs/s)" t_py n_pairs/t_py)

# ── Rust ──────────────────────────────────────────────────────────────────────
println("\nRunning Rust pipeline...")
t_rs = @elapsed begin
    raw_rs, corr_rs, pairs_rs, lags_rs = CCG.compute_all_pairs(
        matrices, frs; jitter_window=JITTER_WIN
    )
end
println(@sprintf "  Rust   : %.2f s  (%.1f pairs/s)" t_rs n_pairs/t_rs)
println(@sprintf "  Speedup: %.1fx" t_py/t_rs)

# ── Numerical comparison ──────────────────────────────────────────────────────
println("\nNumerical comparison (first 10 pairs)...")
ccg_py_real = real.(Matrix{ComplexF64}(ccg_py_out))  # n_pairs × n_lags
corrtime    = Float64.(ccg_py.get_corrtvec(n_time, "corrected"))
min_len     = min(size(ccg_py_real, 2), size(corr_rs, 1))

@printf "%-8s %-15s %-15s %-12s\n" "Pair" "Py peak lag" "Rust peak lag" "Correlation"
println("-"^55)

lags_ms_py = corrtime .* (BINSIZE * 1000)
lags_ms_rs = lags_rs  .* (BINSIZE * 1000)

for k in 1:min(10, n_pairs)
    py_ccg   = ccg_py_real[k, 1:min_len]
    rs_ccg   = corr_rs[1:min_len, k]
    c        = cor(py_ccg, rs_ccg)
    py_peak  = lags_ms_py[argmax(py_ccg)]
    rs_peak  = lags_ms_rs[argmax(rs_ccg)]
    @printf "%-8d %-15.1f %-15.1f %-12.4f\n" k py_peak rs_peak c
end

# ── Save ──────────────────────────────────────────────────────────────────────
h5open("benchmark_real.h5", "w") do f
    f["t_py"]      = t_py
    f["t_rs"]      = t_rs
    f["speedup"]   = t_py / t_rs
    f["n_neurons"] = n_neurons
    f["n_pairs"]   = n_pairs
    f["n_trials"]  = n_trials
    f["n_time"]    = n_time
end
println("\nSaved → benchmark_real.h5")