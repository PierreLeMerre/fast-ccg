using Plots
using HDF5
using Printf
using Statistics
using Random

# ── Configuration ─────────────────────────────────────────────────────────────
const RESULTS_FILE = "ccg_216300_20200518-probe0.h5"
const WIN_MS       = 100.0   # x-axis half-range in ms
const N_PER_PAGE   = 10      # subplots per page

# Which pairs to plot:
#   nothing  → only significant pairs (default)
#   integer  → that many randomly sampled pairs (regardless of significance)
const N_RANDOM = nothing

# ── Load results ──────────────────────────────────────────────────────────────
println("Loading CCG results...")
h5open(RESULTS_FILE, "r") do f
    global raw           = read(f["raw"])
    global jitter        = read(f["jitter"])
    global corrected     = read(f["corrected"])
    global lags_ms       = read(f["lags_ms"])
    global unit_i        = read(f["unit_i"])
    global unit_m        = read(f["unit_m"])
    global is_excitatory = Bool.(read(f["is_excitatory"]))
    global is_inhibitory = Bool.(read(f["is_inhibitory"]))
    global peak_lags     = read(f["peak_lags"])
    global peak_zs       = read(f["peak_zs"])
    global trough_lags   = read(f["trough_lags"])
    global trough_zs     = read(f["trough_zs"])
    global tp            = read(f["tp"])
end

# File contains only significant pairs — all indices are valid
n_pairs = size(raw, 2)
println("Loaded $n_pairs significant pairs  ($(sum(is_excitatory)) excitatory, $(sum(is_inhibitory)) inhibitory)")

# ── Select pairs to plot ──────────────────────────────────────────────────────
plot_idx = isnothing(N_RANDOM) ? collect(1:n_pairs) : sort(randperm(n_pairs)[1:min(N_RANDOM, n_pairs)])
label    = isnothing(N_RANDOM) ? "significant" : "random_$(length(plot_idx))"
println("Plotting $(length(plot_idx)) pairs  ($label)")

# ── Single-pair plot ──────────────────────────────────────────────────────────
const BASELINE_LO = 50.0
const BASELINE_HI = 100.0
const PEAK_WIN    = 10.0

function plot_ccg_pair(k, lags_ms, raw, jitter, corrected, unit_i, unit_m,
                       peak_lags, peak_zs, trough_lags, trough_zs, tp,
                       is_excitatory, is_inhibitory; show_legend=false)
    raw_k = raw[:, k]
    jit_k = jitter[:, k]
    cor_k = corrected[:, k]

    baseline_mask = (abs.(lags_ms) .> BASELINE_LO) .& (abs.(lags_ms) .< BASELINE_HI)
    bl_mean = mean(cor_k[baseline_mask])
    bl_sd   = std(cor_k[baseline_mask])

    type_str = is_excitatory[k] && is_inhibitory[k] ? " EI" :
               is_excitatory[k] ? " E" :
               is_inhibitory[k] ? " I" : ""
    tp_str = is_excitatory[k] ? @sprintf(" TP=%.3f", tp[k]) : ""
    title_str = @sprintf("u%d–u%d  z=%.1f/%+.1f  (%+.1f/%+.1fms)%s%s",
                         unit_i[k], unit_m[k],
                         peak_zs[k], trough_zs[k],
                         peak_lags[k], trough_lags[k], type_str, tp_str)
    cor_color = is_inhibitory[k] && !is_excitatory[k] ? :royalblue : :crimson

    p = plot(
        lags_ms, raw_k;
        label      = "raw",
        color      = :black,
        linewidth  = 1.0,
        fillrange  = 0,
        fillalpha  = 0.10,
        fillcolor  = :black,
        xlabel     = "Lag (ms)",
        ylabel     = "CCG",
        title      = title_str,
        titlefont  = font(7),
        tickfont   = font(6),
        guidefont  = font(7),
        legend     = show_legend ? :topright : false,
        legendfont = font(6),
        xlims      = (-WIN_MS, WIN_MS),
        xticks     = -100:25:100,
        grid       = false,
        framestyle = :box,
    )

    plot!(p, lags_ms, jit_k;
        label     = "jitter",
        color     = :forestgreen,
        linewidth = 0.8,
        linestyle = :dot,
    )
    plot!(p, lags_ms, cor_k;
        label     = "jitter-corrected",
        color     = cor_color,
        linewidth = 1.5,
    )

    # Baseline shading
    vspan!(p, [ BASELINE_LO,  BASELINE_HI]; color=:royalblue, alpha=0.08, label=false)
    vspan!(p, [-BASELINE_HI, -BASELINE_LO]; color=:royalblue, alpha=0.08, label=false)

    # Peak search window
    vspan!(p, [-PEAK_WIN, PEAK_WIN]; color=:orange, alpha=0.08, label=false)

    # Excitatory threshold (7 SD) and inhibitory threshold (-3 SD)
    hline!(p, [bl_mean + 7*bl_sd]; color=:crimson,    linestyle=:dash, linewidth=0.8, label=false)
    hline!(p, [bl_mean - 3*bl_sd]; color=:royalblue,  linestyle=:dash, linewidth=0.8, label=false)

    # Baseline mean
    hline!(p, [bl_mean]; color=:gray, linestyle=:dot, linewidth=0.8, label=false)

    # Zero-lag
    vline!(p, [0.0]; color=:gray, linestyle=:dash, linewidth=0.8, label=false)

    return p
end

# ── Paginated output ──────────────────────────────────────────────────────────
n_pages = ceil(Int, length(plot_idx) / N_PER_PAGE)

for page in 1:n_pages
    idx_range = ((page-1)*N_PER_PAGE + 1) : min(page*N_PER_PAGE, length(plot_idx))
    page_pairs = plot_idx[idx_range]

    plots = [
        plot_ccg_pair(k, lags_ms, raw, jitter, corrected, unit_i, unit_m,
                      peak_lags, peak_zs, trough_lags, trough_zs, tp,
                      is_excitatory, is_inhibitory;
                      show_legend = (k == page_pairs[1]))
        for k in page_pairs
    ]

    # Pad last page
    while length(plots) < N_PER_PAGE
        push!(plots, plot(framestyle=:none, legend=false))
    end

    fig = plot(
        plots...;
        layout         = (5, 2),
        size           = (900, 1100),
        left_margin    = 8Plots.mm,
        bottom_margin  = 6Plots.mm,
        top_margin     = 4Plots.mm,
        plot_title     = "CCG pairs ($label) — page $page/$n_pages",
        plot_titlefont = font(10),
    )

    png_file = @sprintf("ccg_%s_p%02d.png", label, page)
    savefig(fig, png_file)
    println("Saved → $png_file")
end

println("\nDone. $(length(plot_idx)) pairs across $n_pages page(s).")
