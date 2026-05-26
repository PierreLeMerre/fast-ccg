# ── Imports ───────────────────────────────────────────────────────────────────
include("../lib/load_data.jl")
include("../lib/significance.jl")
using PyCall
using HDF5
using Statistics
using Printf

# ── Load Python CCG module ────────────────────────────────────────────────────
# Add the python/ directory to Python's path so we can import the script
pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
const ccg_py = pyimport("ccg_jittercorrected")

# ── Configuration (must match run_ccg.jl exactly for fair comparison) ─────────
const NWB_PATH   = "/Volumes/MorseSSD/NPX_Database/frozen_Aversion_Nat_Submission/216300_20200518-probe0.nwb"
const EPOCH_PATH = "/Volumes/MorseSSD/Pierre_Server_BACKUP/PFCmap_NHP/preprocessing/metrics_extraction/timeselections/timeselections_Carlen/216300_20200518-probe0__TSELprestim3__STATEactive__all.h5"
const BINSIZE    = 0.0005   # 0.5ms
const MIN_FR     = 0.0005
const JITTER_WIN = 25       # Python script uses bins directly (25ms at 1ms = 25, but we use 0.5ms bins so 50)
                             # NOTE: Python jitter() uses timepoints, so pass 50 for 25ms at 0.5ms bins

# ── Load data (same functions as Rust pipeline) ───────────────────────────────
println("Loading spike times...")
spk_times, unit_ids = load_spike_times(NWB_PATH)
println("  $(length(spk_times)) units loaded")

println("Loading epochs...")
ts_in, ts_out = load_epochs(EPOCH_PATH)
println("  $(length(ts_in)) epochs loaded")
println("  Epoch duration: $(round((ts_out[1]-ts_in[1])*1000, digits=1)) ms")

# ── Bin spikes (identical to Rust pipeline) ───────────────────────────────────
println("\nBinning spikes...")
matrices, frs, kept_ids = prepare_all_neurons(
    spk_times, ts_in, ts_out;
    binsize = BINSIZE,
    min_fr  = MIN_FR
)

n_time, n_trials = size(matrices[1])
n_neurons        = length(matrices)
n_pairs          = n_neurons * (n_neurons - 1) ÷ 2

println("  Matrix size per neuron: [$n_time bins × $n_trials trials]")
println("  Total pairs to compute: $n_pairs")

# ── Build spikemat [n_neurons × n_trials × n_time] for Python ────────────────
# Python get_ccgjitter expects: units × trials × timepoints
println("\nBuilding spikemat for Python...")
spikemat = zeros(Float64, n_neurons, n_trials, n_time)
for (i, mat) in enumerate(matrices)
    # mat is [n_time × n_trials] in Julia → permute to [n_trials × n_time]
    spikemat[i, :, :] = permutedims(mat, (2, 1))
end

# ── Run Python CCG ────────────────────────────────────────────────────────────
println("\nComputing CCGs (Python, serial)...")
t_start = time()
ccg_mat = ccg_py.get_ccgjitter(spikemat, JITTER_WIN)
t_elapsed = time() - t_start

println("  Done in $(round(t_elapsed, digits=2))s")
println("  Output size: $(size(ccg_mat))")

# ── Extract results from Python output ───────────────────────────────────────
# Python returns [n_pairs × n_lags] complex array — take real part
# Build lag vector using Python's get_corrtvec
corrtime = ccg_py.get_corrtvec(n_time, "corrected")
lags_ms  = Float64.(corrtime) .* (BINSIZE * 1000)

# ccg_mat from Python is [n_pairs × n_lags]
corrected_py = real.(Matrix{ComplexF64}(ccg_mat))  # n_pairs × n_lags
corrected_py = permutedims(corrected_py, (2, 1))   # → n_lags × n_pairs to match Rust convention

# ── Significance testing (same function as Rust pipeline) ────────────────────
println("\nTesting significance...")
is_sig, peak_lags, peak_zs = test_significance(
    corrected_py, lags_ms;
    peak_window_ms = 10.0,
    baseline_lo_ms = 50.0,
    baseline_hi_ms = 100.0,
    threshold_sd   = 7.0,
    binsize_ms     = BINSIZE * 1000
)

# ── Quick summary ─────────────────────────────────────────────────────────────
println("\nTop 5 pairs by corrected CCG peak:")
println("  Pair      Peak lag (ms)    Peak value")
println("  ─────────────────────────────────────")

peak_vals = [maximum(corrected_py[:, k]) for k in 1:n_pairs]
order     = sortperm(peak_vals, rev=true)

# Build pairs list matching Python's ordering
pairs = [(i, m) for i in 1:n_neurons for m in (i+1):n_neurons]

for rank in 1:min(5, n_pairs)
    k      = order[rank]
    i, m   = pairs[k]
    unit_i = kept_ids[i]
    unit_m = kept_ids[m]
    @printf "  (unit %d, unit %d)   %+.1f ms    %.4f\n" unit_i unit_m peak_lags[k] peak_vals[k]
end

# ── Save results ──────────────────────────────────────────────────────────────
println("\nSaving results...")
h5open("ccg_results_python.h5", "w") do f
    f["corrected"]  = corrected_py
    f["lags_ms"]    = lags_ms
    f["kept_ids"]   = kept_ids
    f["pairs_i"]    = [p[1] for p in pairs]
    f["pairs_m"]    = [p[2] for p in pairs]
    f["is_sig"]     = Float64.(is_sig)
    f["peak_lags"]  = peak_lags
    f["peak_zs"]    = peak_zs
end
println("  Saved to ccg_results_python.h5")