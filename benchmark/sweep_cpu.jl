#=
CPU format sweep: EllSparse SELL-1 / SELL-32 vs stdlib CSC, a threaded CSR loop,
MKL sparse, and dense BLAS, over matrix size × family × operation. Not run in CI.

Run from the repo root (one thread per physical core):

    julia --project=benchmark --startup-file=no -e 'using Pkg; Pkg.instantiate()'   # once
    julia -t 12 --project=benchmark --startup-file=no benchmark/sweep_cpu.jl
    julia -t 12 --project=benchmark --startup-file=no benchmark/sweep_cpu.jl mkl

The `mkl` run is a separate process because MKLSparse.jl overrides `mul!` for
SparseMatrixCSC on load, which would silently turn the stdlib baseline into MKL; that
override is also how the MKL format is measured.
Results go to benchmark/results/cpu.csv (or cpu_mkl.csv); plot with benchmark/plot.jl.
BENCH_SECONDS (default 0.5) sets the sampling budget per measurement.
=#

const MKL = "mkl" in ARGS
if MKL
    ENV["MKL_NUM_THREADS"] = string(Threads.nthreads())
    using MKLSparse
end

using EllSparse
using SparseMatricesCSR

include("sweep_common.jl")

BLAS.set_num_threads(Threads.nthreads())

const MAX_NNZ = parse(Float64, get(ENV, "BENCH_MAX_NNZ", "1.2e8"))
const DENSE_MAX = 10_000      # dense BLAS only up to this many rows
const SERIAL_CUTOFF = 4096    # same serial/threaded switch EllSparse uses

# ── Threaded CSR: the plain row-parallel loop, as a fair multithreaded CSR baseline ──

struct ThreadedCSR{T, M <: SparseMatrixCSR} <: AbstractMatrix{T}
    A::M
end
ThreadedCSR(A::SparseMatrixCSR{1, T}) where {T} = ThreadedCSR{T, typeof(A)}(A)
Base.size(T::ThreadedCSR) = size(T.A)
Base.getindex(T::ThreadedCSR, i::Int, j::Int) = T.A[i, j]

csr(S) = SparseMatrixCSR(transpose(sparse(transpose(S))))   # CSC of Sᵀ = CSR of S

@inline function _csr_row(A, x::AbstractVector, i)
    (rp, cv, nz) = (A.rowptr, A.colval, A.nzval)
    acc = 0.0
    @inbounds @simd for p in rp[i]:(rp[i + 1] - 1)
        acc += nz[p] * x[cv[p]]
    end
    return acc
end

@inline function _csr_row!(Y, A, X, i, α, β)
    for c in axes(X, 2)
        acc = _csr_row(A, view(X, :, c), i)
        @inbounds Y[i, c] = iszero(β) ? α * acc : muladd(β, Y[i, c], α * acc)
    end
    return nothing
end

# Same serial cutoff and :dynamic @threads schedule as EllSparse's own CPU path. For a
# block X each row is applied to every column while it is hot in cache, so the matrix
# streams from memory once per block, not once per column.
function _csr_mul!(Y, T::ThreadedCSR, X, α, β)
    A = T.A
    if size(Y, 1) < SERIAL_CUTOFF
        for i in axes(Y, 1)
            _csr_row!(Y, A, X, i, α, β)
        end
    else
        Threads.@threads for i in axes(Y, 1)
            _csr_row!(Y, A, X, i, α, β)
        end
    end
    return Y
end
LinearAlgebra.mul!(Y::AbstractVector, T::ThreadedCSR, X::AbstractVector, α::Number, β::Number) = _csr_mul!(Y, T, X, α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, T::ThreadedCSR, X::AbstractMatrix, α::Number, β::Number) = _csr_mul!(Y, T, X, α, β)
LinearAlgebra.mul!(Y::AbstractVector, T::ThreadedCSR, X::AbstractVector) = _csr_mul!(Y, T, X, true, false)
LinearAlgebra.mul!(Y::AbstractMatrix, T::ThreadedCSR, X::AbstractMatrix) = _csr_mul!(Y, T, X, true, false)

# ── Formats ──────────────────────────────────────────────────────────────────────
#
# Each entry is (name, matrix, supports_adjoint). The adjoint of a CSR matrix is a scatter
# (each row adds into many outputs), which a threaded loop can only do with atomics or
# per-thread buffers, so threaded CSR skips it rather than fall back to serial.

function formats(S)
    # MKLSparse.jl routes mul! on SparseMatrixCSC (and its transpose) to MKL; its own CSR
    # type is internal. transpose(CSC of Sᵀ) is S in CSR storage, which MKL applies as a
    # row-parallel gather — MKL's fast orientation, the same one SELL uses.
    MKL && return [("MKL CSR", transpose(sparse(transpose(S))), true)]
    A1 = sell(S)
    fs = Any[
        ("SELL-1", A1, true),
        ("SELL-32", reslice(A1, Val(32)), true),
        ("CSC (stdlib)", S, true),
        ("CSR (threaded)", ThreadedCSR(csr(S)), false),
    ]
    size(S, 1) ≤ DENSE_MAX && push!(fs, ("dense BLAS", Matrix(S), true))
    return fs
end

# ── Sweep ────────────────────────────────────────────────────────────────────────

const NAME = MKL ? "cpu_mkl" : "cpu"
const DEVICE = Sys.cpu_info()[1].model
const BASE = (; backend = "cpu", device = DEVICE, threads = Threads.nthreads())

println("device: $DEVICE   threads: $(Threads.nthreads())   BLAS: $(BLAS.get_config().loaded_libs[1].libname)")
println(MKL ? "formats: MKL CSR via MKLSparse.jl" : "formats: SELL-1, SELL-32, CSC, threaded CSR, dense ≤ $DENSE_MAX rows", "\n")

# Threaded triad over 3 × 512 MB arrays, far past the last-level cache
function triad_threaded(N = 2^26)
    (a, b, c) = (zeros(N), rand(N), rand(N))
    f() = Threads.@threads for i in eachindex(a)
        @inbounds a[i] = b[i] + 3.0 * c[i]
    end
    return 24N / timeit(f) / 1.0e9
end

llc_bytes() = parse(Int, split(readlines(`lscpu --caches=NAME,ALL-SIZE --bytes`)[end])[2])

MKL || emit_meta("cpu"; device = DEVICE, threads = Threads.nthreads(), triad_gbs = triad_threaded(),
    llc_bytes = llc_bytes(), julia = string(VERSION))

fresh = Ref(true)
for (fam, Ntarget) in sweep_cases(MAX_NNZ)
    S = int32(fam.build(Ntarget))
    (m, n) = size(S)
    pad32 = sell_padding(S)
    println("$(fam.label): $m rows, nnz $(nnz(S)), SELL-32 padding $(round(100pad32; digits = 1))%")
    meta = (; BASE..., family = fam.key, label = fam.label, m, nnz = nnz(S), pad32)

    x, v = randn(MersenneTwister(3), n), randn(MersenneTwister(4), m)
    (yref, zref) = (ref_mul(S, x), ref_adj(S, v))
    Xs = map(r -> randn(MersenneTwister(5), n, r), NRHS)
    Xrefs = map(X -> reduce(hcat, map(c -> ref_mul(S, X[:, c]), axes(X, 2))), Xs)

    rows = []
    for (name, A, adj) in formats(S)
        y, z = zeros(m), zeros(n)
        calls = MKL ? MKLSparse.__mklsparse_calls_count[] : 0
        mul!(y, A, x)
        MKL && (MKLSparse.__mklsparse_calls_count[] > calls || error("mul! did not reach MKL"))
        relerr(y, yref) < 1.0e-12 || error("$name A*x mismatch: $(relerr(y, yref))")
        push!(rows, row(; meta..., op = "spmv", format = name, useful_bytes = useful_bytes(S),
            time_s = timeit(() -> mul!(y, A, x))))
        report(rows[end])

        if adj
            mul!(z, A', v)
            relerr(z, zref) < 1.0e-12 || error("$name A'x mismatch: $(relerr(z, zref))")
            push!(rows, row(; meta..., op = "spmv_adj", format = name, useful_bytes = useful_bytes(S),
                time_s = timeit(() -> mul!(z, A', v)), deterministic = repeatable(() -> mul!(z, A', v), z)))
            report(rows[end])
        end

        foreach(zip(NRHS, Xs, Xrefs)) do (r, X, Xref)
            Y = zeros(m, r)
            mul!(Y, A, X)
            relerr(Y, Xref) < 1.0e-12 || error("$name A*X (r = $r) mismatch: $(relerr(Y, Xref))")
            push!(rows, row(; meta..., op = "spmm", format = name, nrhs = r, useful_bytes = useful_bytes(S, r),
                time_s = timeit(() -> mul!(Y, A, X))))
            report(rows[end])
        end
    end
    emit!(NAME, rows, fresh)
    GC.gc()
end
println("\nwrote ", joinpath(RESULTS, "$NAME.csv"))
