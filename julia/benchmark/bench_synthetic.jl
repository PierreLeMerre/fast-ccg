include("../lib/load_data.jl")
include("../lib/significance.jl")
include("../lib/ccg.jl")
using .CCG
using PyCall
using Statistics
using Printf
using HDF5
using Random

ENV["RAYON_NUM_THREADS"] = string(Sys.CPU_THREADS)
println("Rayon threads: $(Sys.CPU_THREADS)")

pushfirst!(PyVector(pyimport("sys")."path"),
           joinpath(@__DIR__, "..", "python"))
const ccg_py = pyimport("ccg_jittercorrected")

# ── Parameters ────────────────────────────────────────────────────────────────
const BINSIZE     = 0.0005
const JITTER_WIN  = 50
const N_TRIALS    = 75
const N_TIME_FULL = 2000
const WIN_SZ      = 0.1
const WIN_BINS    = round(Int, WIN_SZ / BINSIZE)  # 200 bins

const N_NEURON_RANGE = [5, 10, 20, 50, 100]

println("="^60)
println("SYNTHETIC BENCHMARK: Python vs Rust CCG")
println("  binsize=$(BINSIZE*1000)ms  trials=$N_TRIALS")
println("  full window=$N_TIME_FULL bins  CCG window=$WIN_BINS bins")
println("="^60)

results = []

for n_neurons in N_NEURON_RANGE
    n_pairs = n_neurons * (n_neurons - 1) ÷ 2
    println("\n── $n_neurons neurons ($n_pairs pairs) ──")

    # Generate synthetic spike matrices on full window
    matrices_full = [Float64.(rand(N_TIME_FULL, N_TRIALS) .< 0.1) for _ in 1:n_neurons]
    frs           = [mean(m) for m in matrices_full]
    matrices_crop = [m[1:WIN_BINS, :] for m in matrices_full]

    # ── Python: full window ────────────────────────────────────────────────
    spikemat_full = zeros(Float64, n_neurons, N_TRIALS, N_TIME_FULL)
    for (i, mat) in enumerate(matrices_full)
        spikemat_full[i, :, :] = permutedims(mat, (2, 1))
    end
    t_py_full = @elapsed begin
        ccg_py.get_ccgjitter(spikemat_full, JITTER_WIN)
    end
    println(@sprintf "  Python (full %d bins) : %.3f s" N_TIME_FULL t_py_full)

    # ── Python: cropped window — fair comparison ───────────────────────────
    spikemat_crop = zeros(Float64, n_neurons, N_TRIALS, WIN_BINS)
    for (i, mat) in enumerate(matrices_crop)
        spikemat_crop[i, :, :] = permutedims(mat, (2, 1))
    end
    t_py_crop = @elapsed begin
        ccg_py.get_ccgjitter(spikemat_crop, JITTER_WIN)
    end
    println(@sprintf "  Python (crop %d bins) : %.3f s" WIN_BINS t_py_crop)

    # ── Rust: full window ──────────────────────────────────────────────────
    t_rs_full = @elapsed begin
        CCG.compute_all_pairs(matrices_full, frs; jitter_window=JITTER_WIN)
    end
    println(@sprintf "  Rust   (full %d bins) : %.3f s" N_TIME_FULL t_rs_full)

    # ── Rust: cropped window ───────────────────────────────────────────────
    t_rs_crop = @elapsed begin
        CCG.compute_all_pairs(matrices_crop, frs; jitter_window=JITTER_WIN)
    end
    println(@sprintf "  Rust   (crop %d bins) : %.3f s" WIN_BINS t_rs_crop)

    println(@sprintf "  Speedup (full,  fair) : %.1fx" t_py_full/t_rs_full)
    println(@sprintf "  Speedup (crop,  fair) : %.1fx" t_py_crop/t_rs_crop)

    push!(results, (
        n_neurons    = n_neurons,
        n_pairs      = n_pairs,
        t_py_full    = t_py_full,
        t_py_crop    = t_py_crop,
        t_rs_full    = t_rs_full,
        t_rs_crop    = t_rs_crop,
        speedup_full = t_py_full / t_rs_full,
        speedup_crop = t_py_crop / t_rs_crop,
    ))
end

# ── Summary table ─────────────────────────────────────────────────────────────
println("\n" * "="^78)
println("SUMMARY")
println("="^78)
@printf "%-10s %-8s %-12s %-12s %-12s %-12s %-10s %-10s\n" "N neurons" "N pairs" "Py full" "Py crop" "Rust full" "Rust crop" "× full" "× crop"
println("-"^78)
for r in results
    @printf "%-10d %-8d %-12.3f %-12.3f %-12.3f %-12.3f %-10.1f %-10.1f\n" r.n_neurons r.n_pairs r.t_py_full r.t_py_crop r.t_rs_full r.t_rs_crop r.speedup_full r.speedup_crop
end

# ── Numerical validation ──────────────────────────────────────────────────────
println("\n── Numerical validation (n=5 neurons, cropped vs cropped) ──")
Random.seed!(42)
n_neurons     = 5
matrices_full = [Float64.(rand(N_TIME_FULL, N_TRIALS) .< 0.1) for _ in 1:n_neurons]
frs           = [mean(m) for m in matrices_full]
matrices_crop = [m[1:WIN_BINS, :] for m in matrices_full]

# Python cropped
spikemat_crop = zeros(Float64, n_neurons, N_TRIALS, WIN_BINS)
for (i, mat) in enumerate(matrices_crop)
    spikemat_crop[i, :, :] = permutedims(mat, (2, 1))
end
ccg_py_out = real.(Matrix{ComplexF64}(ccg_py.get_ccgjitter(spikemat_crop, JITTER_WIN)))
corrtime   = Float64.(ccg_py.get_corrtvec(WIN_BINS, "corrected"))
lags_py_ms = corrtime .* (BINSIZE * 1000)

# Rust cropped
_, corr_rs, _, lags_rs = CCG.compute_all_pairs(matrices_crop, frs; jitter_window=JITTER_WIN)
lags_rs_ms = lags_rs .* (BINSIZE * 1000)

# Compare first pair
min_len   = min(size(ccg_py_out, 2), size(corr_rs, 1))
corr_coef = cor(ccg_py_out[1, 1:min_len], corr_rs[1:min_len, 1])
max_diff  = maximum(abs.(ccg_py_out[1, 1:min_len] .- corr_rs[1:min_len, 1]))
println(@sprintf "  Correlation Python vs Rust : %.6f  (expected ~1.0)" corr_coef)
println(@sprintf "  Max absolute difference    : %.6f" max_diff)

# ── Save ──────────────────────────────────────────────────────────────────────
h5open("benchmark_synthetic.h5", "w") do f
    f["n_neurons"]    = [r.n_neurons    for r in results]
    f["n_pairs"]      = [r.n_pairs      for r in results]
    f["t_py_full"]    = [r.t_py_full    for r in results]
    f["t_py_crop"]    = [r.t_py_crop    for r in results]
    f["t_rs_full"]    = [r.t_rs_full    for r in results]
    f["t_rs_crop"]    = [r.t_rs_crop    for r in results]
    f["speedup_full"] = [r.speedup_full for r in results]
    f["speedup_crop"] = [r.speedup_crop for r in results]
end
println("\nSaved → benchmark_synthetic.h5")