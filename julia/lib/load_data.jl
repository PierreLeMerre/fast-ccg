using HDF5
using Statistics
using Random
using ProgressMeter

"""
    load_spike_times(nwb_path)

Load all unit spike times from an NWB/HDF5 file.

Returns:
  spk_times : Vector{Vector{Float64}} — one vector per unit, timestamps in seconds
  unit_ids  : Vector of unit IDs
"""
function load_spike_times(nwb_path::String)
    h5open(nwb_path, "r") do nwb
        unit_times_data = read(nwb["units/spike_times"])
        unit_times_idx  = read(nwb["units/spike_times_index"])
        unit_ids        = read(nwb["units/id"])

        pushfirst!(unit_times_idx, 1)

        spk_times = [
            unit_times_data[unit_times_idx[i]+1 : unit_times_idx[i+1]]
            for i in 1:length(unit_ids)
        ]

        # Fix first unit edge case
        pushfirst!(spk_times[1], unit_times_data[unit_times_idx[1]])

        return spk_times, unit_ids
    end
end

"""
    firing_rate_hz(spk_times)

Compute mean firing rate in Hz from raw spike timestamps:
  FR = n_spikes / (t_last - t_first)

Returns 0.0 for units with fewer than 2 spikes.
"""
function firing_rate_hz(spk_times::Vector{Float64})
    length(spk_times) < 2 && return 0.0
    return length(spk_times) / (maximum(spk_times) - minimum(spk_times))
end

"""
    chunk_intervals(t_start, t_end; chunk_duration=3.0)

Split [t_start, t_end] into non-overlapping windows of `chunk_duration` seconds.
Any remainder shorter than one full chunk is dropped.

Returns a Matrix{Float64} [n_chunks × 2] with columns [t_in  t_out].

This is the key tool for keeping memory under control when processing a whole
recording: instead of one giant FFT over millions of bins, each chunk becomes
one trial in the [n_time × n_trials] matrix, keeping n_time small.
"""
function chunk_intervals(t_start::Float64, t_end::Float64; chunk_duration::Float64 = 3.0)
    n   = floor(Int, (t_end - t_start) / chunk_duration)
    n == 0 && error("Recording duration $(t_end - t_start)s is shorter than chunk_duration=$(chunk_duration)s")
    ts  = t_start .+ (0:n-1) .* chunk_duration
    return hcat(ts, ts .+ chunk_duration)
end

"""
    whole_recording_intervals(spk_times; chunk_duration=3.0, max_intervals=nothing)

Build intervals covering the whole recording by chunking 0 → t_last into
windows of `chunk_duration` seconds (default 3 s).

If `max_intervals` is an integer, randomly sample that many chunks (without
replacement). Use `nothing` to keep all chunks.

Use `chunk_duration=Inf` to get a single interval — not recommended for long recordings.

Returns a Matrix{Float64} [n_chunks × 2].
"""
function whole_recording_intervals(
    spk_times::Vector{Vector{Float64}};
    chunk_duration::Float64 = 3.0,
    max_intervals::Union{Int, Nothing} = nothing
)
    t_end = maximum(maximum(s) for s in spk_times if !isempty(s))
    isinf(chunk_duration) && return [0.0  t_end]
    ivs = chunk_intervals(0.0, t_end; chunk_duration=chunk_duration)
    if !isnothing(max_intervals) && max_intervals < size(ivs, 1)
        idx = sort(randperm(size(ivs, 1))[1:max_intervals])
        ivs = ivs[idx, :]
    end
    return ivs
end

"""
    bin_spikes_to_matrix(spk_times, intervals; binsize=0.001)

Convert raw spike timestamps into a [n_time × n_trials] spike count matrix.

Arguments:
  spk_times : Vector{Float64} — spike timestamps in seconds for ONE neuron
  intervals : Matrix{Float64} [n_trials × 2], columns [t_in  t_out]
  binsize   : bin size in seconds (default 1ms)

Returns:
  Matrix{Float64} of shape [n_time × n_trials]
"""
function bin_spikes_to_matrix(
    spk_times::Vector{Float64},
    intervals::Matrix{Float64};
    binsize::Float64 = 0.001
)
    n_trials  = size(intervals, 1)
    epoch_dur = intervals[1, 2] - intervals[1, 1]
    n_time    = round(Int, epoch_dur / binsize)

    mat = zeros(Float64, n_time, n_trials)

    for k in 1:n_trials
        t_start = intervals[k, 1]
        t_end   = intervals[k, 2]

        i_start = searchsortedfirst(spk_times, t_start)
        i_end   = searchsortedlast(spk_times, t_end)

        for idx in i_start:i_end
            bin = floor(Int, (spk_times[idx] - t_start) / binsize) + 1
            if 1 <= bin <= n_time
                mat[bin, k] += 1.0
            end
        end
    end

    return mat
end

"""
    prepare_all_neurons(spk_times, unit_ids, intervals; binsize, min_fr_hz, select_units)

Bin neurons into spike matrices and compute firing rates.

Arguments:
  spk_times    : Vector{Vector{Float64}} — one per neuron
  unit_ids     : Vector — NWB unit IDs, one per neuron (from load_spike_times)
  intervals    : Matrix{Float64} [n_trials × 2] — [t_in  t_out] rows
  binsize      : bin size in seconds (default 1ms)
  min_fr_hz    : minimum firing rate in Hz; neurons below this are dropped (default 0 = keep all)
  select_units : optional Vector of unit IDs to restrict to (default nothing = all units)

Returns:
  matrices  : Vector{Matrix{Float64}} — spike matrices, one per kept neuron
  frs_hz    : Vector{Float64} — firing rate in Hz per kept neuron
  kept_ids  : Vector — unit IDs of kept neurons
"""
function prepare_all_neurons(
    spk_times::Vector{Vector{Float64}},
    unit_ids,
    intervals::Matrix{Float64};
    binsize::Float64      = 0.001,
    min_fr_hz::Float64    = 0.0,
    select_units          = nothing
)
    unit_set = isnothing(select_units) ? nothing : Set(Iterators.flatten(
        x isa AbstractRange ? x : (x,) for x in select_units
    ))

    matrices = Matrix{Float64}[]
    frs_hz   = Float64[]
    kept_ids = []

    prog = Progress(length(spk_times); desc="Binning neurons: ", showspeed=true)
    for (i, spk) in enumerate(spk_times)
        uid = unit_ids[i]
        if !isnothing(unit_set) && uid ∉ unit_set
            next!(prog); continue
        end
        fr = firing_rate_hz(spk)
        if fr > min_fr_hz
            mat = bin_spikes_to_matrix(spk, intervals; binsize=binsize)
            push!(matrices, mat)
            push!(frs_hz, fr)
            push!(kept_ids, uid)
        end
        next!(prog)
    end
    finish!(prog)

    n_sel = isnothing(unit_set) ? length(spk_times) : length(unit_set)
    println("Kept $(length(kept_ids)) / $n_sel neurons (min_fr=$(min_fr_hz) Hz)")
    return matrices, frs_hz, kept_ids
end

"""
    crop_to_window(matrices; win_bins)

Crop [n_time × n_trials] spike matrices to the first win_bins bins.
"""
function crop_to_window(
    matrices::Vector{Matrix{Float64}};
    n_time_full::Int,
    win_bins::Int
)
    return [m[1:win_bins, :] for m in matrices]
end
