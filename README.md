# ripccg

**Fast, jitter-corrected cross-correlograms for neural spike trains — written in Rust, callable from Julia.**

`ripccg` computes pairwise cross-correlograms (CCGs) across all neuron pairs in a population recording, with jitter correction following the Siegle et al. methodology. It is designed to be called from Julia via a C FFI interface, and achieves a ~9× speedup over a pure-Python equivalent.

---

## Features

- **FFT-based cross-correlation** via [`rustfft`](https://crates.io/crates/rustfft) — zero-copy inner loop with pre-allocated buffers and pre-computed plans
- **Jitter correction** — subtracts the jittered predictor and normalizes by geometric mean firing rate and triangular overlap, yielding a corrected CCG in units of coincidence rate
- **Parallel pair computation** via [`rayon`](https://crates.io/crates/rayon) — all N×(N-1)/2 pairs processed concurrently across CPU cores
- **C FFI interface** — exposes `ccg_pair_ffi` and `compute_all_pairs_ffi` as `extern "C"` symbols, callable from Julia with `ccall`
- **Diagnostic entry points** — `diagnose_ffi`, `diagnose_ccg_ffi`, `diagnose_ccg_pair_ffi` for validating the FFI bridge during development

---

## Algorithm

For each neuron pair (i, m):

1. Bin spike trains into a `[n_time × n_trials]` matrix
2. Compute the raw CCG as the mean trial-averaged FFT cross-correlation
3. Jitter each spike train within windows of `jitter_window` bins, preserving the PSTH, to produce the jitter predictor
4. Compute the jitter CCG with the same procedure
5. Output the **corrected CCG**:

```
corrected[lag] = (raw[lag] - jitter[lag]) / (√(FR_i × FR_m) × θ[lag])
```

where `θ[lag]` is the triangular overlap normalization.

The output for each pair is `[raw | corrected]` concatenated — a vector of length `2 × (2 × n_time − 1)`.

---

## Performance

Benchmarked on NHP Neuropixels recordings (probe 0, active state epochs):

| Neurons | Pairs | Time   | Throughput     |
|---------|-------|--------|----------------|
| ~120    | ~7140 | ~0.75s | ~9500 pairs/s  |

Compared to a Python reference implementation: **~9.4× faster**.

---

## Building

```bash
cargo build --release
```

Produces `target/release/libcross_correlogram.dylib` (macOS) or `.so` (Linux).

---

## Julia usage

```julia
include("lib/ccg.jl")
using .CCG

# matrices: Vector of [n_time × n_trials] Float64 matrices, one per neuron
# frs:      Vector of mean firing rates (spikes/bin)
raw, corrected, pairs, lags = CCG.compute_all_pairs(matrices, frs; jitter_window=50)
```

The Julia wrapper handles pointer passing, memory layout (column-major → row-major transposition), and output parsing. See `run_ccg.jl` for a full pipeline example loading from NWB and `__TSEL__` epoch files.

### Significance testing

```julia
include("lib/significance.jl")
is_sig, peak_lags_ms, peak_zs = test_significance(
    corrected, lags_ms;
    peak_window_ms  = 10.0,
    baseline_lo_ms  = 50.0,
    baseline_hi_ms  = 100.0,
    threshold_sd    = 7.0,
    binsize_ms      = 0.5
)
```

Significance is assessed against a baseline window (50–100 ms), with a 7 SD threshold applied within ±10 ms of lag 0.

---

## Data inputs

The pipeline is designed for NHP Neuropixels recordings:

- **Spike data**: loaded from NWB files via HDF5.jl
- **Epochs**: loaded from `__TSEL__` HDF5 files (Carlen Lab format), providing trial onset/offset times

---

## Crate structure

```
src/
├── lib.rs      — FFI entry points and diagnostic functions
├── xcorr.rs    — FFT cross-correlation engine (FftPlans, xcorr_fft_planned, make_theta, make_target)
└── jitter.rs   — Jitter correction, mean_xcorr_fast, ccg_pair, compute_all_pairs
```

---

## Dependencies

| Crate         | Version | Role                              |
|---------------|---------|-----------------------------------|
| `rustfft`     | 6.2     | FFT engine                        |
| `ndarray`     | 0.16    | N-dimensional arrays              |
| `rayon`       | 1.10    | Data-parallel pair iteration      |
| `num-complex` | 0.4     | Complex number type for FFT buffers |

---

## Tests

```bash
cargo test
```

Covers: self-correlation peak at lag 0, jitter rate preservation, offset train peak lag recovery, and parallel pair count correctness.

---

## License

MIT
