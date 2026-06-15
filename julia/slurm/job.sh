#!/bin/bash
#SBATCH --job-name=ccg
#SBATCH --time=02:00:00          # wall time — adjust per recording length
#SBATCH --mem=32G                # RAM — large NWB files with many units need more
#SBATCH --cpus-per-task=8        # Rust uses rayon; match to available cores
#SBATCH --partition=normal       # change to your cluster's partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ── Arguments (set by submit_jobs.jl via sbatch positional args) ──────────────
NWB_PATH="$1"
OUT_DIR="$2"
SYSIMAGE="$3"

# ── Repo root (two levels up from this script) ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$REPO_ROOT/julia/rust/run_ccg_job.jl"

# ── Load Julia (adjust module name for your cluster) ─────────────────────────
# module load julia/1.10.0

# ── Build Rust library if not present ────────────────────────────────────────
RUST_LIB="$REPO_ROOT/target/release/libcross_correlogram.so"
if [ ! -f "$RUST_LIB" ]; then
    echo "Rust library not found — building..."
    cd "$REPO_ROOT" && cargo build --release
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== CCG job ==="
echo "NWB     : $NWB_PATH"
echo "Output  : $OUT_DIR"
echo "Sysimage: $SYSIMAGE"
echo "Julia   : $(julia --version)"
echo "Started : $(date)"
echo "Node    : $(hostname)"
echo "==============="

if [ -f "$SYSIMAGE" ]; then
    julia --sysimage "$SYSIMAGE" --threads "$SLURM_CPUS_PER_TASK" "$JOB_SCRIPT" "$NWB_PATH" "$OUT_DIR"
else
    echo "WARNING: sysimage not found — starting without it (slow compile on first run)"
    julia --threads "$SLURM_CPUS_PER_TASK" "$JOB_SCRIPT" "$NWB_PATH" "$OUT_DIR"
fi

echo "Finished: $(date)"
