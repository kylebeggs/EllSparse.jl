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
using CUDA.CUSPARSE
using CUDA.CUSPARSE: cusparseSpMatDescr_t, cusparseCreateSlicedEll, cusparseDestroySpMat,
    cusparseSpMV_bufferSize, cusparseSpMV_preprocess, cusparseSpMV, CuSparseMatrixDescriptor,
    CuDenseVectorDescriptor, handle
using LinearAlgebra
using Random: MersenneTwister
using SparseArrays

CUDA.functional() || error("CUDA is not functional on this machine")

# ── cuSPARSE SELL view of a device SellMatrix{T, 32} ──────────────────────────────
#
# Both libraries store a slice column-major (slot-major), so the buffers line up. With a
# one-based index base cuSPARSE reads EllSparse's `sliceptr` and values as-is; only the
# padding sentinel differs (EllSparse 0, cuSPARSE -1), so `colind` is the one copy.

mutable struct CuSell{T, V, I}
    handle::cusparseSpMatDescr_t
    vals::V   # kept alive: cuSPARSE holds raw pointers into these
    colind::I
    sliceptr::I
end

# CUDA.jl's generated binding types these device buffers as host `Ptr{Cvoid}`, so pass
# the raw device address explicitly (the arrays are kept alive by the CuSell struct).
devptr(a) = Ptr{Cvoid}(UInt(pointer(a)))

function CuSell(A::SellMatrix{T, C}) where {T, C}
    st = structure(A)
    colind = map(c -> iszero(c) ? -one(c) : c, st.colind)   # padding: 0 → -1
    vals, sliceptr = parent(A), st.sliceptr
    nnz = count(!iszero, st.colind)
    ref = Ref{cusparseSpMatDescr_t}()
    GC.@preserve vals colind sliceptr cusparseCreateSlicedEll(
        ref, size(A)..., nnz, length(vals), C,
        devptr(sliceptr), devptr(colind), devptr(vals),
        eltype(sliceptr), eltype(colind), 'O', T,
    )
    M = CuSell{T, typeof(vals), typeof(colind)}(ref[], vals, colind, sliceptr)
    finalizer(M -> cusparseDestroySpMat(M.handle), M)
    return M
end
# lets `ccall` take a CuSell directly, which roots it (and its buffers) for the call
Base.unsafe_convert(::Type{cusparseSpMatDescr_t}, M::CuSell) = M.handle

# A reusable y ← op(A) x: descriptors, workspace, and (optional) preprocessing done once.
function spmv_plan(descA, trans, x, y, alg)
    T = eltype(y)
    α, β = Ref{T}(one(T)), Ref{T}(zero(T))
    dx, dy = CuDenseVectorDescriptor(x), CuDenseVectorDescriptor(y)
    sz = Ref{Csize_t}()
    cusparseSpMV_bufferSize(handle(), trans, α, descA, dx, β, dy, T, alg, sz)
    buf = CuVector{UInt8}(undef, max(sz[], 1))
    cusparseSpMV_preprocess(handle(), trans, α, descA, dx, β, dy, T, alg, buf)
    return () -> cusparseSpMV(handle(), trans, α, descA, dx, β, dy, T, alg, buf)
end

# ── Test matrices ─────────────────────────────────────────────────────────────────

laplacian1d(n) = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))

function laplacian2d(n)
    L, Id = laplacian1d(n), sparse(1.0I, n, n)
    return kron(L, Id) + kron(Id, L)                 # 5-point, n² rows
end

function stencil27(n)
    B = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
    S = kron(B, kron(B, B))                         # 27-point, n³ rows
    nonzeros(S) .= randn(MersenneTwister(1), nnz(S))
    return S
end

# Meshless-operator shape: k nearest neighbors in index space, a local but non-banded
# gather — the structure an RBF-FD weight matrix has after a locality-preserving ordering.
function knn_stencil(N, k)
    rng = MersenneTwister(2026)
    I_ = repeat(1:N; inner = k)
    J = [clamp(i + rand(rng, -4k:4k), 1, N) for i in 1:N for _ in 1:k]
    return sparse(I_, J, randn(rng, N * k), N, N)
end

# Uniformly random columns: no locality at all, the worst case for the x gather.
function random_stencil(N, k)
    rng = MersenneTwister(2026)
    return sparse(repeat(1:N; inner = k), rand(rng, 1:N, N * k), randn(rng, N * k), N, N)
end

# Irregular rows: per-row lengths vary 1..2k, where per-slice padding starts to cost.
function ragged(N, k)
    rng = MersenneTwister(7)
    lens = rand(rng, 1:(2k), N)
    I_ = reduce(vcat, map(fill, 1:N, lens))
    J = [clamp(i + rand(rng, -4k:4k), 1, N) for (i, l) in enumerate(lens) for _ in 1:l]
    return sparse(I_, J, randn(rng, length(I_)), N, N)
end

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
