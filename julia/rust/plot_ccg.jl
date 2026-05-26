using Plots
using HDF5
using Printf
using Statistics

const BINSIZE = 0.0005
const WIN_SZ  = 0.1

# ── Load results ──────────────────────────────────────────────────────────────
println("Loading CCG results...")
h5open("ccg_results.h5", "r") do f
    global raw       = read(f["raw"])
    global corrected = read(f["corrected"])
    global lags_ms   = read(f["lags_ms"])
    global kept_ids  = read(f["kept_ids"])
    global pairs_i   = read(f["pairs_i"])
    global pairs_m   = read(f["pairs_m"])
    global is_sig    = Bool.(read(f["is_sig"]))
    global peak_lags = read(f["peak_lags"])
    global peak_zs   = read(f["peak_zs"])
end

pairs   = collect(zip(pairs_i, pairs_m))
n_pairs = size(raw, 2)
sig_idx = findall(is_sig)

println("Significant pairs: $(length(sig_idx)) / $n_pairs")

if isempty(sig_idx)
    println("No significant pairs found — check threshold or data.")
    exit()
end

# ── Plot significant pairs, 10 per page ───────────────────────────────────────
# Baseline shading region limits
baseline_lo = 50.0
baseline_hi = 100.0
peak_window = 10.0

n_per_page = 10
n_pages    = ceil(Int, length(sig_idx) / n_per_page)

for page in 1:n_pages
    idx_range = ((page-1)*n_per_page + 1) : min(page*n_per_page, length(sig_idx))
    page_idx  = sig_idx[idx_range]

    plots = []

    for k in page_idx
        i, m   = pairs[k]
        unit_i = kept_ids[i]
        unit_m = kept_ids[m]

        raw_k = raw[:, k]
        cor_k = corrected[:, k]

        # Baseline stats for annotation
        baseline_mask = (abs.(lags_ms) .> baseline_lo) .& (abs.(lags_ms) .< baseline_hi)
        bl_mean = mean(cor_k[baseline_mask])
        bl_sd   = std(cor_k[baseline_mask])
        z_score = peak_zs[k]
        p_lag   = peak_lags[k]

        p = plot(
            lags_ms, raw_k,
            label      = "raw",
            color      = :black,
            linewidth  = 1.2,
            fillrange  = 0,
            fillalpha  = 0.12,
            fillcolor  = :black,
            xlabel     = "Lag (ms)",
            ylabel     = "CCG",
            title      = @sprintf("u%d–u%d  peak=%+.1fms  z=%.1f", 
                                   unit_i, unit_m, p_lag, z_score),
            titlefont  = font(7),
            tickfont   = font(6),
            guidefont  = font(7),
            legend     = k == page_idx[1] ? :topright : false,
            legendfont = font(6),
            xlims      = (-WIN_SZ * 1000, WIN_SZ * 1000),
            xticks     = -100:25:100,
            grid       = false,
            framestyle = :box,
        )

        # Jitter-corrected CCG
        plot!(p,
            lags_ms, cor_k,
            label     = "jitter-corrected",
            color     = :crimson,
            linewidth = 1.5,
        )

        # Shade baseline regions (both sides)
        vspan!(p, [baseline_lo, baseline_hi],
            color = :royalblue, alpha = 0.08, label = false)
        vspan!(p, [-baseline_hi, -baseline_lo],
            color = :royalblue, alpha = 0.08, label = false)

        # Shade peak search window
        vspan!(p, [-peak_window, peak_window],
            color = :orange, alpha = 0.08, label = false)

        # 7 SD threshold line
        hline!(p, [bl_mean + 7 * bl_sd],
            color     = :crimson,
            linestyle = :dash,
            linewidth = 0.8,
            label     = false,
        )

        # Baseline mean
        hline!(p, [bl_mean],
            color     = :gray,
            linestyle = :dot,
            linewidth = 0.8,
            label     = false,
        )

        # Zero lag reference
        vline!(p, [0.0],
            color     = :gray,
            linestyle = :dash,
            linewidth = 0.8,
            label     = false,
        )

        push!(plots, p)
    end

    # Pad to fill grid if last page has fewer than 10
    while length(plots) < n_per_page
        push!(plots, plot(framestyle=:none, legend=false))
    end

    fig = plot(
        plots...,
        layout        = (5, 2),
        size          = (900, 1100),
        left_margin   = 8Plots.mm,
        bottom_margin = 6Plots.mm,
        top_margin    = 4Plots.mm,
        plot_title    = "Significant CCG pairs — page $page/$n_pages",
        plot_titlefont = font(10),
    )

    png_file = @sprintf("ccg_significant_p%02d.png", page)
    pdf_file = @sprintf("ccg_significant_p%02d.pdf", page)
    savefig(fig, png_file)
    savefig(fig, pdf_file)
    println("Saved → $png_file  $pdf_file")
end

println("\nDone. $(length(sig_idx)) significant pairs across $n_pages page(s).")
