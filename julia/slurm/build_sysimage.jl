# Build a Julia sysimage that pre-compiles all CCG job dependencies.
# Run ONCE on the cluster login node before submitting jobs:
#
#   julia build_sysimage.jl
#
# This produces ccg_sysimage.so in the same directory (~5-10 min first run).
# Jobs then load it with:  julia --sysimage ccg_sysimage.so run_ccg_job.jl ...
#
# Prerequisites (run once in your Julia environment):
#   import Pkg
#   Pkg.add(["PackageCompiler", "HDF5", "ProgressMeter"])

using PackageCompiler

sysimage_path      = joinpath(@__DIR__, "ccg_sysimage.so")
precompile_file    = joinpath(@__DIR__, "precompile_exec.jl")

println("Building sysimage → $sysimage_path")
println("This takes 5-15 minutes...")

create_sysimage(
    [:HDF5, :Statistics, :Printf, :Random, :ProgressMeter],
    sysimage_path           = sysimage_path,
    precompile_execution_file = precompile_file,
)

println("Done. Sysimage saved to: $sysimage_path")
