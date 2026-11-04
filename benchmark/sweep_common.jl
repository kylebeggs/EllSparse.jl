#=
Shared machinery for sweep_cpu.jl and sweep_gpu.jl: timing, the correctness
reference, and CSV output.
=#

using BenchmarkTools
using CSV
using DataFrames
using LinearAlgebra
using SparseArrays

include("matrices.jl")

const NRHS = (8, 32)       # SpMM block widths
const SECONDS = parse(Float64, get(ENV, "BENCH_SECONDS", "0.5"))
const RESULTS = joinpath(@__DIR__, "results")

call(f) = f()

# Minimum wall time of `f()` in seconds. `sync(f)` makes the call (e.g. under CUDA.@sync).
function timeit(f; sync = call)
    b = @benchmarkable $sync($f) seconds = SECONDS   # tune! uses this budget too
    tune!(b)
    return minimum(run(b)).time / 1.0e9
end

# Bytes a product must move at minimum, the same for every format: one Float64 value and
# one Int32 index per stored nonzero, plus x and y once per right-hand side. Padding,
# re-reads, and Int64 indices all show up as lower effective bandwidth, not more bytes.
useful_bytes(S, r = 1) = 12 * nnz(S) + 8 * r * (size(S, 1) + size(S, 2))

# Serial CSC loops, independent of any `mul!` method another package may override.
function ref_mul(S, x)
    y = zeros(size(S, 1))
    (rv, nz) = (rowvals(S), nonzeros(S))
    for j in axes(S, 2), p in nzrange(S, j)
        y[rv[p]] += nz[p] * x[j]
    end
    return y
end

function ref_adj(S, v)
    (rv, nz) = (rowvals(S), nonzeros(S))
    return map(j -> sum(p -> nz[p] * v[rv[p]], nzrange(S, j); init = 0.0), axes(S, 2))
end

relerr(a, b) = norm(Array(a) - b) / norm(b)

# Runs a product `reps` times; true when every result is bitwise identical to the first.
function repeatable(f, out; reps = 20)
    f()
    ref = Array(out)
    return all(_ -> (f(); Array(out) == ref), 1:reps)
end

row(; kwargs...) = (; backend = "", device = "", threads = 0, family = "", label = "",
    m = 0, nnz = 0, pad32 = 0.0, op = "", format = "", nrhs = 1, time_s = NaN,
    useful_bytes = 0, deterministic = missing, kwargs...)

# Appends rows to `results/<name>.csv`, writing the header on first use in this run.
function emit!(name, rows, fresh::Ref{Bool})
    mkpath(RESULTS)
    path = joinpath(RESULTS, "$name.csv")
    CSV.write(path, DataFrame(rows); append = !fresh[])
    fresh[] = false
    return path
end

function report(r)
    gbs = r.useful_bytes / r.time_s / 1.0e9
    det = ismissing(r.deterministic) ? "" : (r.deterministic ? "  repeatable" : "  NOT repeatable")
    return println("  ", rpad(r.op * (r.nrhs > 1 ? "×$(r.nrhs)" : ""), 10), rpad(r.format, 22),
        lpad(round(r.time_s * 1.0e6; digits = 1), 11), " μs  ", lpad(round(gbs; digits = 1), 7), " GB/s", det)
end

sell_padding(S) = 1 - nnz(S) / length(parent(sell(S; slice_height = 32, transpose_map = false)))

# Writes one row of machine facts the plots draw as reference lines: measured triad
# bandwidth (the streaming roof) and last-level cache size (where the matrix stops fitting).
function emit_meta(name; kwargs...)
    mkpath(RESULTS)
    return CSV.write(joinpath(RESULTS, "$(name)_meta.csv"), DataFrame([(; kwargs...)]))
end

# STREAM-style triad a = b + s c: reads two arrays and writes one, 24 bytes per element.
triad_gbs(a, b, c; sync = call) = 24 * length(a) / timeit(() -> (a .= b .+ 3.0 .* c); sync) / 1.0e9
