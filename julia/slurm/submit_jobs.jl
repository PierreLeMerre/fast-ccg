# Submit one SLURM job per NWB file.
#
# Usage:
#   julia submit_jobs.jl <nwb_dir> <out_dir> [--dry-run]
#
# Arguments:
#   nwb_dir  — directory containing .nwb files (searched recursively)
#   out_dir  — directory where per-file ccg_*.h5 results will be written
#   --dry-run  — print sbatch commands without submitting
#
# Example:
#   julia submit_jobs.jl /data/nwb/ /results/ccg/

# ── Parse arguments ───────────────────────────────────────────────────────────
length(ARGS) >= 2 || error("Usage: julia submit_jobs.jl <nwb_dir> <out_dir> [--dry-run]")

nwb_dir  = ARGS[1]
out_dir  = ARGS[2]
dry_run  = "--dry-run" in ARGS

isdir(nwb_dir) || error("NWB directory not found: $nwb_dir")
isdir(out_dir) || mkpath(out_dir)

# ── Locate files ──────────────────────────────────────────────────────────────
nwb_files = String[]
for (root, dirs, files) in walkdir(nwb_dir)
    for f in files
        endswith(f, ".nwb") && push!(nwb_files, joinpath(root, f))
    end
end

isempty(nwb_files) && error("No .nwb files found under $nwb_dir")
println("Found $(length(nwb_files)) NWB file(s)")

# ── Paths relative to this script ─────────────────────────────────────────────
script_dir   = @__DIR__
job_script   = joinpath(script_dir, "job.sh")
sysimage     = joinpath(script_dir, "ccg_sysimage.so")

isfile(job_script) || error("Job template not found: $job_script")
isfile(sysimage)   || @warn "Sysimage not found at $sysimage — jobs will start without it (slow). Run build_sysimage.jl first."

# ── Submit ────────────────────────────────────────────────────────────────────
submitted = 0
for nwb in nwb_files
    stem     = splitext(basename(nwb))[1]
    log_out  = joinpath(out_dir, "$(stem).out")
    log_err  = joinpath(out_dir, "$(stem).err")

    cmd = `sbatch
        --job-name=ccg_$(stem)
        --output=$(log_out)
        --error=$(log_err)
        $(job_script) $(nwb) $(out_dir) $(sysimage)`

    if dry_run
        println("[dry-run] ", join(cmd.exec, " "))
    else
        result = readchomp(`$cmd`)
        println("Submitted $stem → $result")
        submitted += 1
    end
end

dry_run || println("\nSubmitted $submitted / $(length(nwb_files)) jobs.")
