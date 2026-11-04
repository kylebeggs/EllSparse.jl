#=
Manual GPU benchmark: EllSparse SELL-32 kernels vs cuSPARSE SpMV on the same matrices,
with cuSPARSE reading EllSparse's own buffers through its Sliced-ELL (SELL) format.
Not run in CI — this needs real CUDA hardware.

Run from the repo root on a CUDA machine:

    julia --project=benchmark --startup-file=no -e 'using Pkg; Pkg.instantiate()'   # once
    julia --project=benchmark --startup-file=no benchmark/cusparse.jl

Every cuSPARSE call reuses a preallocated workspace and a descriptor built once, so the
timings compare kernels, not setup. Results are documented in docs/src/cusparse.md.
=#

using EllSparse
using Adapt
using BenchmarkTools
using CUDA

CUDA.functional() || error("CUDA is not functional on this machine")

include("cusell.jl")      # CuSell, spmv_plan
include("matrices.jl")    # test-matrix generators

cases = [
    ("2D Laplacian, 5-pt (1000²)", laplacian2d(1000)),
    ("3D 27-pt stencil (64³)", stencil27(64)),
    ("kNN stencil, k = 32 (1M)", knn_stencil(1_000_000, 32)),
    ("random columns, k = 32 (1M)", random_stencil(1_000_000, 32)),
    ("ragged rows, 1–64 (1M)", ragged(1_000_000, 32)),
]

# ── Run ───────────────────────────────────────────────────────────────────────────

const SELL_ALG = CUSPARSE.CUSPARSE_SPMV_SELL_ALG1
const CSR_ALG = CUSPARSE.CUSPARSE_SPMV_CSR_ALG1

bench(f) = minimum(run(@benchmarkable CUDA.@sync $f(); seconds = 3)).time

println("device: ", CUDA.name(CUDA.device()), "   cuSPARSE ", CUSPARSE.version())
println("Float64; minimum time per product (μs)\n")

μs(x) = round(x / 1.0e3; digits = 1)

# Run a product `reps` times and report whether every result is bitwise identical.
function deterministic(f, out; reps = 20)
    f(); ref = Array(out)
    return all(_ -> (f(); Array(out) == ref), 1:reps)
end

rows = map(cases) do (label, S)
    m, n = size(S)
    A = adapt(CuArray, sell(S; slice_height = 32))
    sellA, csrA = CuSell(A), CuSparseMatrixDescriptor(CuSparseMatrixCSR(S), 'O')

    x, v = CuArray(randn(MersenneTwister(3), n)), CuArray(randn(MersenneTwister(4), m))
    y = (ell = CUDA.zeros(Float64, m), sell = CUDA.zeros(Float64, m), csr = CUDA.zeros(Float64, m))
    z = (ell = CUDA.zeros(Float64, n), sell = CUDA.zeros(Float64, n), csr = CUDA.zeros(Float64, n))

    fwd = (
        ell = () -> mul!(y.ell, A, x),
        sell = spmv_plan(sellA, 'N', x, y.sell, SELL_ALG),
        csr = spmv_plan(csrA, 'N', x, y.csr, CSR_ALG),
    )
    adj = (
        ell = () -> mul!(z.ell, A', v),
        sell = spmv_plan(sellA, 'T', v, z.sell, SELL_ALG),
        csr = spmv_plan(csrA, 'T', v, z.csr, CSR_ALG),
    )

    # correctness first: every path must agree with the host CSC product
    foreach(f -> f(), fwd); foreach(f -> f(), adj)
    yref, zref = S * Array(x), S' * Array(v)
    rel(a, b) = norm(Array(a) - b) / norm(b)
    errs = (map(a -> rel(a, yref), y)..., map(a -> rel(a, zref), z)...)
    maximum(errs) < 1.0e-12 || error("$label: mismatch vs host, rel errs = $errs")

    det = map((f, o) -> deterministic(f, o), adj, z)
    padding = 1 - nnz(S) / length(parent(A))
    tf, ta = map(bench, fwd), map(bench, adj)
    println(rpad(label, 30), " nnz/row ", rpad(round(nnz(S) / m; digits = 1), 5),
        " pad ", rpad(string(round(100padding; digits = 1), "%"), 6),
        " | A*x  EllSparse ", rpad(μs(tf.ell), 7), " SELL ", rpad(μs(tf.sell), 7), " CSR ", rpad(μs(tf.csr), 7),
        " | A'x  EllSparse ", rpad(μs(ta.ell), 7), " SELL ", rpad(μs(ta.sell), 7), " CSR ", rpad(μs(ta.csr), 7),
        " | bitwise-repeatable A'x: ", det)
    (; label, k = nnz(S) / m, padding, tf, ta, det)
end

# Markdown tables for docs/src/cusparse.md
mark(t, ok) = ok ? "$(μs(t))" : "$(μs(t))†"
println("\n| Matrix | nnz/row | SELL padding | EllSparse | cuSPARSE SELL | cuSPARSE CSR |")
println("|:--|--:|--:|--:|--:|--:|")
foreach(r -> println("| $(r.label) | $(round(r.k; digits = 1)) | $(round(100r.padding; digits = 1))% | $(μs(r.tf.ell)) | $(μs(r.tf.sell)) | $(μs(r.tf.csr)) |"), rows)
println("\n| Matrix | EllSparse | cuSPARSE SELL | cuSPARSE CSR |")
println("|:--|--:|--:|--:|")
foreach(r -> println("| $(r.label) | $(mark(r.ta.ell, r.det.ell)) | $(mark(r.ta.sell, r.det.sell)) | $(mark(r.ta.csr, r.det.csr)) |"), rows)
println("\n† results not bitwise-repeatable across 20 runs")
