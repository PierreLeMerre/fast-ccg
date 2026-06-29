module CCG

using Libdl
using Statistics

nextpow2_julia(n::Int) = (p = 1; while p < n; p <<= 1; end; p)

const LIB_PATH = joinpath(@__DIR__, "..", "..", "target", "release", "libcross_correlogram.$(Libdl.dlext)")
const lib = Libdl.dlopen(LIB_PATH)

"""
    ccg_pair(spikes_i, spikes_m, fr_i, fr_m; jitter_window=25)

Compute raw, jitter and jitter-corrected cross-correlogram for one neuron pair.

Arguments:
  spikes_i, spikes_m : [n_time × n_trials] Float64 binary spike matrices
  fr_i, fr_m         : mean firing rates in spikes/bin (use mean(spikes))
  jitter_window      : jitter window in bins (default 25)

Returns:
  raw       : Vector{Float64} length 2*n_time-1, mean coincidences per trial
  jitter    : Vector{Float64} length 2*n_time-1, jitter predictor (mean coincidences per trial)
  corrected : Vector{Float64} length 2*n_time-1, dimensionless normalised excess
  lags      : Vector{Int}, lag values in bins
"""
function ccg_pair(
    spikes_i::Matrix{Float64},
    spikes_m::Matrix{Float64},
    fr_i::Float64,
    fr_m::Float64;
    jitter_window::Int = 25
)
    n_time, n_trials = size(spikes_i)
    ccg_len = 2 * n_time - 1
    nfft    = nextpow2_julia(2 * n_time)

    out = Vector{Float64}(undef, 3 * ccg_len)

    ccall(
        Libdl.dlsym(lib, :ccg_pair_ffi), Cvoid,
        (Ptr{Float64}, Csize_t, Csize_t,
         Ptr{Float64}, Csize_t, Csize_t,
         Cdouble, Cdouble,
         Csize_t,
         Ptr{Float64}, Csize_t),
        spikes_i, n_time, n_trials,
        spikes_m, n_time, n_trials,
        fr_i, fr_m,
        jitter_window,
        out, 3 * ccg_len
    )

    raw       = out[1:ccg_len]
    jitter    = out[ccg_len+1:2*ccg_len]
    corrected = out[2*ccg_len+1:end]
    lags      = collect(-(n_time-1):(n_time-1))
    return raw, jitter, corrected, lags
end

"""
    compute_all_pairs(spikes, firing_rates; jitter_window=25, n_threads=0)

Compute raw, jitter and jitter-corrected CCG for all neuron pairs in parallel.

Arguments:
  spikes        : Vector of [n_time × n_trials] matrices, one per neuron
  firing_rates  : Vector of mean firing rates (spikes/bin), one per neuron
  jitter_window : jitter window in bins (default 25)
  n_threads     : number of Rayon worker threads (0 = all available cores)

Returns:
  raw       : Matrix{Float64} [ccg_len × n_pairs] — mean coincidences per trial
  jitter    : Matrix{Float64} [ccg_len × n_pairs] — jitter predictor (mean coincidences per trial)
  corrected : Matrix{Float64} [ccg_len × n_pairs] — dimensionless normalised excess
  pairs     : Vector of (i, m) tuples — neuron indices for each column
  lags      : Vector{Int} — lag values in bins
"""
function compute_all_pairs(
    spikes::Vector{Matrix{Float64}},
    firing_rates::Vector{Float64};
    jitter_window::Int = 25,
    n_threads::Int     = 0
)
    n_neurons        = length(spikes)
    n_time, n_trials = size(spikes[1])
    ccg_len          = 2 * n_time - 1
    n_pairs          = n_neurons * (n_neurons - 1) ÷ 2
    nfft             = nextpow2_julia(2 * n_time)

    spikes_flat = reduce(hcat, [vec(s) for s in spikes])
    spikes_flat = Float64.(spikes_flat)

    out = Matrix{Float64}(undef, 3 * ccg_len, n_pairs)

    ccall(
        Libdl.dlsym(lib, :compute_all_pairs_ffi), Cvoid,
        (Ptr{Float64}, Csize_t, Csize_t, Csize_t,
         Ptr{Float64},
         Csize_t,
         Csize_t,
         Ptr{Float64}, Csize_t),
        spikes_flat, n_time, n_trials, n_neurons,
        firing_rates,
        jitter_window,
        n_threads,
        out, 3 * ccg_len * n_pairs
    )

    raw       = out[1:ccg_len, :]
    jitter    = out[ccg_len+1:2*ccg_len, :]
    corrected = out[2*ccg_len+1:end, :]
    pairs     = [(i, m) for i in 1:n_neurons for m in (i+1):n_neurons]
    lags      = collect(-(n_time-1):(n_time-1))
    return raw, jitter, corrected, pairs, lags
end

end # module
