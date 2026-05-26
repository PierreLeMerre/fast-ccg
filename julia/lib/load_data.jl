using HDF5
using Statistics

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

        # Fix first unit edge case from your original code
        pushfirst!(spk_times[1], unit_times_data[unit_times_idx[1]])

        return spk_times, unit_ids
    end
end

"""
    load_epochs(epoch_path)

Load epoch start/end timestamps from HDF5 epoch file.

Returns:
  ts_in  : Vector{Float64} — epoch start times in seconds
  ts_out : Vector{Float64} — epoch end times in seconds
"""
function load_epochs(epoch_path::String)
    h5open(epoch_path, "r") do f
        tints  = read(f["tints"])
        ts_in  = tints[1, :]
        ts_out = tints[2, :]
        return Float64.(ts_in), Float64.(ts_out)
    end
end

"""
    bin_spikes_to_matrix(spk_times, ts_in, ts_out; binsize=0.001)

Convert raw spike timestamps into a [n_time × n_trials] spike count matrix.

Each trial is one epoch [ts_in[k], ts_out[k]].
Spikes are binned at `binsize` resolution (default 1ms).
Spike counts > 1 per bin are preserved.

Arguments:
  spk_times : Vector{Float64} — spike timestamps in seconds for ONE neuron
  ts_in     : Vector{Float64} — epoch start times
  ts_out    : Vector{Float64} — epoch end times
  binsize   : bin size in seconds (default 1ms)

Returns:
  Matrix{Float64} of shape [n_time × n_trials]
"""
function bin_spikes_to_matrix(
    spk_times::Vector{Float64},
    ts_in::Vector{Float64},
    ts_out::Vector{Float64};
    binsize::Float64 = 0.001
)
    n_trials = length(ts_in)

    # Compute n_time from first epoch duration (all epochs same length)
    epoch_dur = ts_out[1] - ts_in[1]
    n_time    = round(Int, epoch_dur / binsize)

    mat = zeros(Float64, n_time, n_trials)

    for (k, (t_start, t_end)) in enumerate(zip(ts_in, ts_out))
        # Find spikes within this epoch using searchsorted (fast binary search)
        # equivalent to: spk_times[(spk_times .>= t_start) .& (spk_times .< t_end)]
        i_start = searchsortedfirst(spk_times, t_start)
        i_end   = searchsortedlast(spk_times, t_end)

        for idx in i_start:i_end
            t_spike = spk_times[idx]
            # Convert to bin index (1-based)
            bin = floor(Int, (t_spike - t_start) / binsize) + 1
            if 1 <= bin <= n_time
                mat[bin, k] += 1.0
            end
        end
    end

    return mat
end

"""
    prepare_all_neurons(spk_times, ts_in, ts_out; binsize=0.001, min_fr=0.0)

Bin all neurons into spike matrices and compute firing rates.
Optionally filter out neurons below a minimum firing rate.

Arguments:
  spk_times : Vector{Vector{Float64}} — one per neuron
  ts_in     : Vector{Float64} — epoch starts
  ts_out    : Vector{Float64} — epoch ends
  binsize   : bin size in seconds
  min_fr    : minimum firing rate in spikes/bin to include neuron (default 0 = keep all)

Returns:
  matrices  : Vector{Matrix{Float64}} — spike matrices, one per neuron
  frs       : Vector{Float64} — mean firing rate in spikes/bin
  kept_ids  : Vector{Int} — indices of neurons that passed the fr filter
"""
function prepare_all_neurons(
    spk_times::Vector{Vector{Float64}},
    ts_in::Vector{Float64},
    ts_out::Vector{Float64};
    binsize::Float64 = 0.001,
    min_fr::Float64  = 0.0
)
    matrices = Matrix{Float64}[]
    frs      = Float64[]
    kept_ids = Int[]

    for (i, spk) in enumerate(spk_times)
        mat = bin_spikes_to_matrix(spk, ts_in, ts_out; binsize=binsize)
        fr  = mean(mat)

        if fr > min_fr
            push!(matrices, mat)
            push!(frs, fr)
            push!(kept_ids, i)
        end
    end

    println("Kept $(length(kept_ids)) / $(length(spk_times)) neurons (min_fr=$min_fr spikes/bin)")
    return matrices, frs, kept_ids
end

"""
    crop_to_window(matrices, binsize; win_sz=0.1)

Crop [n_time × n_trials] spike matrices to a ±win_sz window around epoch start.
Reduces n_time from epoch_duration/binsize to 2*win_sz/binsize.
This dramatically reduces FFT size and speeds up computation.
"""
function crop_to_window(
    matrices::Vector{Matrix{Float64}};
    n_time_full::Int,
    win_bins::Int
)
    # Take only first win_bins bins (already aligned to epoch start)
    return [m[1:win_bins, :] for m in matrices]
end