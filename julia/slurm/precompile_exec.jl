# Executed during sysimage compilation to trace package usage.
# Keep lightweight — no actual data needed, just load and exercise each package.
using HDF5
using Statistics
using Printf
using Random
using ProgressMeter

# Exercise HDF5 to force method compilation
buf = IOBuffer()
h5open(buf, "w") do f
    f["x"] = collect(1.0:10.0)
end

# Exercise Statistics
x = randn(100)
_ = mean(x); _ = std(x)

# Exercise ProgressMeter
p = Progress(3; enabled=false)
for _ in 1:3; next!(p); end
