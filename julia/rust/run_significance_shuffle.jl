using HDF5
using Statistics
using Plots
using Printf
using ColorBrewer
using Glob
using Graphs
using GraphRecipes

include("../lib/load_data.jl")   # load_spike_times, firing_rate_hz

filelist = glob("/Volumes/MorseSSD/shuffle_ccg/*_ShuffleCCG.h5")



# ── Config ────────────────────────────────────────────────────────────────────
const IN_PATH  = "/Volumes/MorseSSD/shuffle_ccg/225758_20200519-probe0_ShuffleCCG.h5"
const NWB_DIR  = "/Volumes/MorseSSD/NPX_Database/frozen_Aversion_Nat_Submission"
const PLOTS_DIR = "/Volumes/MorseSSD/shuffle_ccg/Plots"

const THRESHOLD_SD     = 7.0   # excitatory
const NEG_THRESHOLD_SD = 5.0   # inhibitory
const ZERO_LAG_MS      = 0.8   # exclude lags within this of zero
const PEAK_WINDOW_MS   = 10.0  # look for peak/trough within ± this

# ── Load shuffle-CCG data ──────────────────────────────────────────────────────
# HDF5.jl reads dimensions reversed relative to Python/h5py (no implicit
# transpose), so e.g. Python's (185,185,100) comes back as (100,185,185).
# Permute back to the Python shape: corr → (n_units, n_units, n_lags),
# rand → (n_shuffle, n_units, n_units, n_lags).
bin_edges, corr_raw, rand_raw = h5open(IN_PATH, "r") do f
    f["bins"][:], f["correlograms"][:, :, :], f["rand_correlograms"][:, :, :, :]
end
corr = permutedims(corr_raw, (3, 2, 1))
rand = permutedims(rand_raw, (4, 3, 2, 1))

n_units = size(corr, 1)
lags_ms = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2   # bin centers

# ── Spike counts from the matching NWB (for transmission probability) ───────
# The shuffle-CCG file now stores the NWB unit ID for every correlogram
# row/column directly (`unit_ids`), so pairs are matched by ID rather than
# guessed via a firing-rate filter.
recording_name = replace(basename(IN_PATH), r"_[Ss]huffle_?[Cc][Cc][Gg]\.h5$" => "")
nwb_path       = joinpath(NWB_DIR, recording_name * ".nwb")
isfile(nwb_path) || error("No matching NWB file found: $nwb_path")

ccg_unit_ids = h5open(IN_PATH, "r") do f
    Int.(f["unit_ids"][:])
end
length(ccg_unit_ids) == n_units ||
    error("unit_ids length $(length(ccg_unit_ids)) does not match n_units=$n_units in $IN_PATH")

println("Loading spike times from $nwb_path ...")
spk_times, nwb_unit_ids = load_spike_times(nwb_path)
nwb_unit_ids = Int.(nwb_unit_ids)

id_to_idx = Dict(uid => idx for (idx, uid) in enumerate(nwb_unit_ids))
missing_ids = [uid for uid in ccg_unit_ids if !haskey(id_to_idx, uid)]
isempty(missing_ids) || error("Unit ID(s) $missing_ids from $IN_PATH not found in $nwb_path")

n_spikes = [length(spk_times[id_to_idx[uid]]) for uid in ccg_unit_ids]
println("  Matched $n_units units by ID — NWB ids $(ccg_unit_ids[1])..$(ccg_unit_ids[end])")

# ── Per-bin shuffle null (mean/SD across the 40 shuffles) ───────────────────
shuffle_mean = dropdims(mean(Float64.(rand); dims=1); dims=1)
shuffle_std  = dropdims(std(Float64.(rand);  dims=1); dims=1)

excit_mask = (lags_ms .>  ZERO_LAG_MS) .& (lags_ms .<= PEAK_WINDOW_MS)
inhib_mask = (lags_ms .>  ZERO_LAG_MS) .& (lags_ms .<= PEAK_WINDOW_MS)
central_idx = argmin(abs.(lags_ms))   # bin nearest zero lag, used for the artifact check below

# ── Test every pair ───────────────────────────────────────────────────────────
println("unit_i  unit_j  type        peak_lag_ms  z       tp")

n_ex = 0
n_in = 0

sig_pairs = Vector{NamedTuple{(:i,:j,:kind,:lag,:z,:tp),Tuple{Int,Int,String,Float64,Float64,Float64}}}()
EI_mtrx = fill(0.0,n_units,n_units)
Adj_mtrx = fill(0.0,n_units,n_units)

for i in 1:n_units, j in (i+1):n_units
    raw = Float64.(corr[i, j, :])
    mu  = shuffle_mean[i, j, :]
    sd  = shuffle_std[i, j, :]
    z   = zeros(Float64, length(raw))
    valid = sd .> 1e-10
    z[valid] = (raw[valid] .- mu[valid]) ./ sd[valid]

    # Skip pairs where the central (zero-lag) bin itself is extreme — almost
    # certainly a spike-sorting/electrical artifact rather than a real interaction.
    if z[central_idx] > THRESHOLD_SD || z[central_idx] < -NEG_THRESHOLD_SD
        continue
    end

    # excitatory: ≥2 consecutive bins above threshold, in (0.8, 10] ms
    ex_region = z[excit_mask]
    Adj_mtrx[i,j] = sum(ex_region)
    ex_cross  = ex_region .> THRESHOLD_SD
    is_ex     = any(ex_cross[1:end-1] .& ex_cross[2:end])
    if is_ex
        global n_ex += 1
        EI_mtrx[i,j] = 1
        peak_idx = argmax(ex_region)
        peak_z   = ex_region[peak_idx]
        peak_lag = lags_ms[excit_mask][peak_idx]
        # TP = Σ_{τ∈(0.8,10]ms} (raw[τ]-shuffle_mean[τ]) / n_spikes_i
        # (no n_trials factor: these correlograms are whole-recording totals)
        tp_val = sum(raw[excit_mask] .- mu[excit_mask]) / n_spikes[i]
        println("$i\t$j\tEXCITATORY\t$peak_lag\t$peak_z\t$tp_val")
        push!(sig_pairs, (i=i, j=j, kind="EXCITATORY", lag=peak_lag, z=peak_z, tp=tp_val))
    end

    # inhibitory: ≥2 consecutive bins below -threshold, in ±(0.8, 10] ms
    in_region = z[inhib_mask]
    in_cross  = in_region .< -NEG_THRESHOLD_SD
    is_in     = any(in_cross[1:end-1] .& in_cross[2:end])
    if is_in
        global n_in += 1
        EI_mtrx[i,j] = -1
        trough_idx = argmin(in_region)
        trough_z   = in_region[trough_idx]
        trough_lag = lags_ms[inhib_mask][trough_idx]
        # Suppression magnitude — same formula as TP but over the inhibitory window
        sp_val = sum(raw[inhib_mask] .- mu[inhib_mask]) / n_spikes[i]
        println("$i\t$j\tINHIBITORY\t$trough_lag\t$trough_z\t$sp_val")
        push!(sig_pairs, (i=i, j=j, kind="INHIBITORY", lag=trough_lag, z=trough_z, tp=sp_val))
    end
end

n_pairs = n_units * (n_units - 1) ÷ 2
println("\nExcitatory: $n_ex / $n_pairs")
println("Inhibitory: $n_in / $n_pairs")

# ── Plot every significant pair, save into PLOTS_DIR/<recording_name>/ ──────
if !isempty(sig_pairs)
    out_dir = joinpath(PLOTS_DIR, recording_name)
    mkpath(out_dir)
    println("\nSaving $(length(sig_pairs)) plots → $out_dir")

    for p in sig_pairs
        raw = corr[p.i, p.j, :]
        mu  = shuffle_mean[p.i, p.j, :]
        sd  = shuffle_std[p.i, p.j, :]
        color = p.kind == "EXCITATORY" ? :crimson : :royalblue

        plt = plot(
            lags_ms, mu .+ sd; fillrange = mu .- sd, fillalpha = 0.15,
            fillcolor = :gray, linealpha = 0, label = false,
        )
        plot!(plt, lags_ms, mu; color = :gray, linestyle = :dash, linewidth = 0.8, label = "shuffle mean")
        plot!(plt, lags_ms, raw; color = color, linewidth = 1.5, label = "raw")
        vline!(plt, [0.8, -0.8]; color = :black, linestyle = :dot, linewidth = 0.6, label = false)

        tp_str = p.kind == "EXCITATORY" ? @sprintf("  TP=%.3f", p.tp) : ""
        title!(plt, @sprintf("u%d–u%d  %s  z=%.1f @ %.2fms%s", p.i, p.j, p.kind, p.z, p.lag, tp_str);
               titlefont = font(9))
        xlabel!(plt, "lag (ms)")
        ylabel!(plt, "count")

        fname = @sprintf("u%03d_u%03d_%s.png", p.i, p.j, p.kind)
        savefig(plt, joinpath(out_dir, fname))
    end
    println("Done plotting.")
else
    println("\nNo significant pairs — nothing to plot.")
end

h1 = heatmap(EI_mtrx .+ EI_mtrx',
        xlabel = "units",
        ylabel = "units",
        color = :bwr,
        size=(800, 800)
        )

#c = cgrad([:pink,:white,:green], categorical = false)

h2 = heatmap(Adj_mtrx .+ Adj_mtrx',
        xlabel = "units",
        ylabel = "units",
        size=(800, 800),
        clims=(-50, 50),
        colorbar_title = "Σ z-score (0.8–10ms)",
        title = "Adjacency matrix - "* "225758_20200519-probe0",
        color = :magma
        )

savefig(h2,PLOTS_DIR * "Adjacency_matrix_" * "225758_20200519-probe0.svg")		

# ── Scatter: every E/I connection, marker radius ∝ |TP|, color = E/I ────────
ex_pts = filter(p -> p.kind == "EXCITATORY", sig_pairs)
in_pts = filter(p -> p.kind == "INHIBITORY", sig_pairs)
tp_radius(tp) = 3.0 + sqrt.(300.0 * abs(tp))

h3 = plot(xlabel = "unit i", ylabel = "unit j", title = "E/I connections (radius ∝ TP) - "* "225758_20200519-probe0",
          legend = :outertopright, aspect_ratio = :equal, size=(800, 800),
          xlims = (1, n_units), ylims = (1, n_units))
if !isempty(ex_pts)
    scatter!(h3, [p.i for p in ex_pts], [p.j for p in ex_pts];
             markersize = tp_radius.([p.tp for p in ex_pts]),
             color = :crimson, markerstrokewidth = 0, alpha = 0.6, label = "excitatory")
end
if !isempty(in_pts)
    scatter!(h3, [p.i for p in in_pts], [p.j for p in in_pts];
             markersize = tp_radius.([p.tp for p in in_pts]),
             color = :royalblue, markerstrokewidth = 0, alpha = 0.6, label = "inhibitory")
end

savefig(h3,PLOTS_DIR * "E-I_connections_" * "225758_20200519-probe0.svg")		


#plt = plot(h2, h3, layout = Plots.grid(2,1), size = (800, 1300))

matrix_dir = joinpath(PLOTS_DIR, recording_name)
mkpath(matrix_dir)
savefig(plt, joinpath(matrix_dir, "matrices_and_scatter.png"))
println("Saved matrices + scatter → $(joinpath(matrix_dir, "matrices_and_scatter.png"))")

# ── Putative connection graphs: best 2 connected components ──────────────────
# Directed edge i→j for every significant pair (i leads j, same convention as
# TP). "Best" = largest connected component (most units), ties broken by edge
# count — these are the components most likely to show real circuit structure.
G = SimpleDiGraph(n_units)
edge_kind = Dict{Tuple{Int,Int}, String}()
edge_tp   = Dict{Tuple{Int,Int}, Float64}()
for p in sig_pairs
    add_edge!(G, p.i, p.j)
    edge_kind[(p.i, p.j)] = p.kind
    edge_tp[(p.i, p.j)]   = p.tp
end

components = filter(c -> length(c) > 1, weakly_connected_components(G))
sort!(components; by = c -> (length(c), sum(has_edge(G, a, b) for a in c, b in c)), rev = true)

n_graphs = min(2, length(components))
println("\nFound $(length(components)) connected component(s) with ≥2 units — plotting top $n_graphs")

graph_plots = Plots.Plot[]

for (rank, comp) in enumerate(components[1:n_graphs])
    nodes = sort(comp)
    sub, vmap = induced_subgraph(G, nodes)        # vmap[k] = original unit index (row in corr)
    sub_ids   = ccg_unit_ids[vmap]                # NWB unit ID per subgraph node, for labels

    edge_colors = Dict{Tuple{Int,Int}, Symbol}()
    edge_widths = Dict{Tuple{Int,Int}, Float64}()
    for e in edges(sub)
        s, d = src(e), dst(e)
        kind = edge_kind[(vmap[s], vmap[d])]
        edge_colors[(s, d)] = kind == "EXCITATORY" ? :orange : :indigo
        edge_widths[(s, d)] = 1.0 + 15.0 * sqrt(abs(edge_tp[(vmap[s], vmap[d])]))  # scale for visibility
    end

    g = graphplot(sub;
        names        = string.(sub_ids),
        nodeshape    = :circle,
        nodesize     = 0.15,
        nodecolor    = :lightgray,
        markerstrokecolor = :black,
        edgecolor    = edge_colors,
        edgewidth    = edge_widths,
        arrow        = arrow(:closed, :head, 0.4, 0.4),
        curves       = false,
        fontsize     = 7,
        title        = "Putative connections — component $rank ($(length(nodes)) units)  orange=E  indigo=I",
        size         = (700, 700),
    )

    fname = joinpath(matrix_dir, "connected_graph_$(rank).png")
    savefig(g, fname)
    println("Saved → $fname")
    push!(graph_plots, g)
end

# ── Summary plot: matrices + scatter + the 2 best connection graphs ─────────
blank_plot(msg) = plot(framestyle = :none, legend = false, title = msg, titlefont = font(9))
graph1 = length(graph_plots) >= 1 ? graph_plots[1] : blank_plot("no connected component")
graph2 = length(graph_plots) >= 2 ? graph_plots[2] : blank_plot("no 2nd connected component")

summary_plt = plot(h2, h3, graph1, graph2, layout = Plots.grid(2,2), size = (1600, 1300))
savefig(summary_plt, joinpath(matrix_dir, "summary.png"))
savefig(summary_plt, joinpath(matrix_dir, "summary.svg"))
println("Saved summary → $(joinpath(matrix_dir, "summary.png"))")