using Plots
using HDF5
using Printf

# ── Load synthetic benchmark results ─────────────────────────────────────────
println("Loading benchmark results...")
h5open("benchmark_synthetic.h5", "r") do f
    global n_neurons    = read(f["n_neurons"])
    global n_pairs      = read(f["n_pairs"])
    global t_py_full    = read(f["t_py_full"])
    global t_py_crop    = read(f["t_py_crop"])
    global t_rs_full    = read(f["t_rs_full"])
    global t_rs_crop    = read(f["t_rs_crop"])
    global speedup_full = read(f["speedup_full"])
    global speedup_crop = read(f["speedup_crop"])
end

# ── Plot 1: wall time (log scale) ─────────────────────────────────────────────
p1 = plot(
    n_neurons, t_py_full,
    label="Python (full 1s)", color=:steelblue,
    linestyle=:dash, marker=:circle, markersize=5, linewidth=2,
    xlabel="Number of neurons", ylabel="Wall time (s)",
    title="Computation time", titlefont=font(10),
    legend=:topleft, grid=false, framestyle=:box,
    yscale=:log10,
)
plot!(p1, n_neurons, t_py_crop,
    label="Python (±100ms)", color=:steelblue,
    marker=:circle, markersize=5, linewidth=2)
plot!(p1, n_neurons, t_rs_full,
    label="Rust (full 1s)", color=:crimson,
    linestyle=:dash, marker=:circle, markersize=5, linewidth=2)
plot!(p1, n_neurons, t_rs_crop,
    label="Rust (±100ms)", color=:crimson,
    marker=:circle, markersize=5, linewidth=2)

# ── Plot 2: speedup ───────────────────────────────────────────────────────────
p2 = plot(
    n_neurons, speedup_full,
    label="Full window", color=:steelblue,
    marker=:circle, markersize=5, linewidth=2,
    xlabel="Number of neurons", ylabel="Speedup (×)",
    title="Rust speedup over Python", titlefont=font(10),
    legend=:bottomright, grid=false, framestyle=:box,
)
plot!(p2, n_neurons, speedup_crop,
    label="±100ms window", color=:crimson,
    marker=:circle, markersize=5, linewidth=2)
hline!(p2, [1.0],
    color=:gray, linestyle=:dash, linewidth=0.8, label=false)
# Annotate peak speedup
max_idx = argmax(speedup_crop)
annotate!(p2, n_neurons[max_idx], speedup_crop[max_idx] + 0.5,
    text(@sprintf("%.1f×", speedup_crop[max_idx]), 8, :center, :crimson))

# ── Plot 3: throughput (pairs/second) ─────────────────────────────────────────
p3 = plot(
    n_neurons, n_pairs ./ t_py_crop,
    label="Python (±100ms)", color=:steelblue,
    marker=:circle, markersize=5, linewidth=2,
    xlabel="Number of neurons", ylabel="Pairs / second",
    title="Throughput (±100ms window)", titlefont=font(10),
    legend=:topleft, grid=false, framestyle=:box,
)
plot!(p3, n_neurons, n_pairs ./ t_rs_crop,
    label="Rust (±100ms)", color=:crimson,
    marker=:circle, markersize=5, linewidth=2)

# ── Plot 4: real data results if available ────────────────────────────────────
real_path = "benchmark_real.h5"
if isfile(real_path)
    h5open(real_path, "r") do f
        global r_n_neu   = read(f["n_neurons"])
        global r_n_pr    = read(f["n_pairs"])
        global r_t_py    = read(f["t_py"])
        global r_t_rs    = read(f["t_rs"])
        global r_speedup = read(f["speedup"])
        global r_n_tr    = read(f["n_trials"])
        global r_n_ti    = read(f["n_time"])
    end

    p4 = bar(
        ["Python", "Rust"],
        [r_t_py, r_t_rs],
        color      = [:steelblue, :crimson],
        alpha      = 0.8,
        legend     = false,
        ylabel     = "Wall time (s)",
        title      = @sprintf("Real data: %d neurons, %d pairs\n%d trials × %d bins",
                               r_n_neu, r_n_pr, r_n_tr, r_n_ti),
        titlefont  = font(9),
        grid       = false,
        framestyle = :box,
    )
    annotate!(p4, 1, r_t_py + r_t_py*0.03,
        text(@sprintf("%.2fs", r_t_py), 8, :center))
    annotate!(p4, 2, r_t_rs + r_t_py*0.03,
        text(@sprintf("%.2fs\n(%.1f×)", r_t_rs, r_speedup), 8, :center, :crimson))
else
    # Placeholder if real data not yet run
    p4 = plot(
        framestyle = :box, grid = false,
        title = "Real data\n(run bench_real.jl)",
        titlefont = font(10),
        legend = false,
    )
    annotate!(p4, 0.5, 0.5, text("No real data yet", 10, :center, :gray))
end

# ── Assemble ──────────────────────────────────────────────────────────────────
fig = plot(
    p1, p2, p3, p4,
    layout         = (2, 2),
    size           = (950, 750),
    left_margin    = 8Plots.mm,
    bottom_margin  = 8Plots.mm,
    top_margin     = 4Plots.mm,
    plot_title     = "CCG Benchmark: Python vs Rust  |  $(Sys.CPU_THREADS) cores  |  $(Sys.MACHINE)",
    plot_titlefont = font(10),
)

savefig(fig, "benchmark_results.png")
savefig(fig, "benchmark_results.pdf")
println("Saved → benchmark_results.png / .pdf")

# ── Print summary ─────────────────────────────────────────────────────────────
println("\nSynthetic benchmark key numbers:")
println("  N neurons   Py ±100ms    Rust ±100ms   Speedup")
println("  " * "─"^48)
for i in eachindex(n_neurons)
    @printf "  %-10d %-12.3f %-12.3f %.1fx\n" n_neurons[i] t_py_crop[i] t_rs_crop[i] speedup_crop[i]
end

if isfile(real_path)
    println("\nReal data:")
    @printf "  %d neurons, %d pairs\n" r_n_neu r_n_pr
    @printf "  Python : %.2f s  (%.0f pairs/s)\n" r_t_py r_n_pr/r_t_py
    @printf "  Rust   : %.2f s  (%.0f pairs/s)\n" r_t_rs r_n_pr/r_t_rs
    @printf "  Speedup: %.1fx\n" r_speedup
end

display(fig)