#=
CPU sweep: EllSparse vs the sparse-matrix options a Julia user can install today, over
matrix size × family × operation. Not run in CI.

Every package is called the way its own documentation shows. Operations a package has
no specialized method for are skipped rather than timed through LinearAlgebra's generic
dense fallback (which is O(m·n) and would not finish at these sizes).

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

using CompressedSparseBlocks
using EllSparse
using GenericSparseArrays
using KernelAbstractions: KernelAbstractions
using SparseMatricesCSR
using SuiteSparseGraphBLAS
using ThreadedSparseArrays

include("sweep_common.jl")

BLAS.set_num_threads(Threads.nthreads())
SuiteSparseGraphBLAS.gbset(:nthreads, Threads.nthreads())
CompressedSparseBlocks.setWorkers(Threads.nthreads())

# Guard against type piracy: no package loaded above may take over stdlib CSC's mul!
# (MKLSparse does, which is why it gets its own process).
let owner = which(mul!, Tuple{Vector{Float64}, SparseMatrixCSC{Float64, Int32}, Vector{Float64}, Bool, Bool}).module
    MKL || owner ∈ (LinearAlgebra, SparseArrays) || error("SparseMatrixCSC mul! is owned by $owner")
end

const MAX_NNZ = parse(Float64, get(ENV, "BENCH_MAX_NNZ", "1.2e8"))
const DENSE_MAX = 10_000      # dense BLAS only up to this many rows
const SERIAL_CUTOFF = 4096    # same serial/threaded switch EllSparse uses

# ── Reference: a hand-written row-parallel CSR loop (not a package) ──────────────────

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
# Each format is a name, a figure group, and one closure per operation (`nothing` where
# the package has no specialized method). Groups: "julia" for pure-Julia packages,
# "libs" for wrapped C/vendor libraries and the two references.

fmt(name, group; fwd, adj = nothing, spmm = nothing) = (; name, group, fwd, adj, spmm)

const KA_CPU = KernelAbstractions.CPU()
kasync(f) = (f(); KernelAbstractions.synchronize(KA_CPU))   # GenericSparseArrays' mul! doesn't sync

function formats(S)
    # MKLSparse.jl routes mul! on SparseMatrixCSC (and its transpose) to MKL. transpose(CSC
    # of Sᵀ) is S in CSR storage, which MKL applies as a row-parallel gather.
    if MKL
        M = transpose(sparse(transpose(S)))
        return [fmt("MKLSparse", "libs"; fwd = (y, x) -> mul!(y, M, x),
            adj = (z, v) -> mul!(z, M', v), spmm = (Y, X) -> mul!(Y, M, X))]
    end
    A = sell(S)
    C = csr(S)
    R = ThreadedCSR(C)
    # ThreadedSparseArrays threads A'x and A*X (over columns); its README's route to a
    # threaded forward product is to store Sᵀ and apply its adjoint
    T = ThreadedSparseMatrixCSC(S)
    Tt = ThreadedSparseMatrixCSC(sparse(transpose(S)))
    Gr = GenericSparseMatrixCSR(S)
    Gc = GenericSparseMatrixCSC(S)
    B = GBMatrix(S)
    Cb = SparseMatrixCSB(SparseMatrixCSC{Float64, UInt32}(S))   # CSB takes UInt32 or Int64
    fs = Any[
        fmt("EllSparse", "both"; fwd = (y, x) -> mul!(y, A, x), adj = (z, v) -> mul!(z, A', v),
            spmm = (Y, X) -> mul!(Y, A, X)),
        fmt("SparseArrays CSC", "both"; fwd = (y, x) -> mul!(y, S, x), adj = (z, v) -> mul!(z, S', v),
            spmm = (Y, X) -> mul!(Y, S, X)),
        fmt("SparseMatricesCSR", "julia"; fwd = (y, x) -> mul!(y, C, x)),
        fmt("ThreadedSparseArrays", "julia"; fwd = (y, x) -> mul!(y, Tt', x),
            adj = (z, v) -> mul!(z, T', v), spmm = (Y, X) -> mul!(Y, T, X)),
        fmt("GenericSparseArrays CSR", "julia"; fwd = (y, x) -> kasync(() -> mul!(y, Gr, x)),
            adj = (z, v) -> kasync(() -> mul!(z, Gr', v)), spmm = (Y, X) -> kasync(() -> mul!(Y, Gr, X))),
        fmt("GenericSparseArrays CSC", "julia"; fwd = (y, x) -> kasync(() -> mul!(y, Gc, x)),
            adj = (z, v) -> kasync(() -> mul!(z, Gc', v)), spmm = (Y, X) -> kasync(() -> mul!(Y, Gc, X))),
        fmt("SuiteSparseGraphBLAS", "libs"; fwd = (y, x) -> mul!(y, B, x), adj = (z, v) -> mul!(z, B', v),
            spmm = (Y, X) -> mul!(Y, B, X)),
        fmt("CompressedSparseBlocks", "libs"; fwd = (y, x) -> mul!(y, Cb, x), adj = (z, v) -> mul!(z, Cb', v),
            spmm = (Y, X) -> mul!(Y, Cb, X)),
        fmt("CSR loop (reference)", "libs"; fwd = (y, x) -> mul!(y, R, x), spmm = (Y, X) -> mul!(Y, R, X)),
    ]
    if size(S, 1) ≤ DENSE_MAX
        D = Matrix(S)
        push!(fs, fmt("dense BLAS", "libs"; fwd = (y, x) -> mul!(y, D, x), adj = (z, v) -> mul!(z, D', v),
            spmm = (Y, X) -> mul!(Y, D, X)))
    end
    return fs
end

# ── Sweep ────────────────────────────────────────────────────────────────────────

const NAME = MKL ? "cpu_mkl" : "cpu"
const DEVICE = Sys.cpu_info()[1].model
const BASE = (; backend = "cpu", device = DEVICE, threads = Threads.nthreads())

println("device: $DEVICE   threads: $(Threads.nthreads())   BLAS: $(BLAS.get_config().loaded_libs[1].libname)")
println(MKL ? "formats: MKLSparse.jl" : "formats: EllSparse + Julia ecosystem; dense ≤ $DENSE_MAX rows", "\n")

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
    record!(r) = (push!(rows, r); report(r))
    for f in formats(S)
        meta_f = (; meta..., format = f.name, group = f.group)
        y, z = zeros(m), zeros(n)
        calls = MKL ? MKLSparse.__mklsparse_calls_count[] : 0
        f.fwd(y, x)
        MKL && (MKLSparse.__mklsparse_calls_count[] > calls || error("mul! did not reach MKL"))
        relerr(y, yref) < 1.0e-12 || error("$(f.name) A*x mismatch: $(relerr(y, yref))")
        record!(row(; meta_f..., op = "spmv", useful_bytes = useful_bytes(S), time_s = timeit(() -> f.fwd(y, x))))

        if !isnothing(f.adj)
            f.adj(z, v)
            relerr(z, zref) < 1.0e-12 || error("$(f.name) A'x mismatch: $(relerr(z, zref))")
            record!(row(; meta_f..., op = "spmv_adj", useful_bytes = useful_bytes(S),
                time_s = timeit(() -> f.adj(z, v)), deterministic = repeatable(() -> f.adj(z, v), z)))
        end

        isnothing(f.spmm) && continue
        foreach(zip(NRHS, Xs, Xrefs)) do (r, X, Xref)
            Y = zeros(m, r)
            f.spmm(Y, X)
            relerr(Y, Xref) < 1.0e-12 || error("$(f.name) A*X (r = $r) mismatch: $(relerr(Y, Xref))")
            record!(row(; meta_f..., op = "spmm", nrhs = r, useful_bytes = useful_bytes(S, r),
                time_s = timeit(() -> f.spmm(Y, X))))
        end
    end
    emit!(NAME, rows, fresh)
    GC.gc()
end
println("\nwrote ", joinpath(RESULTS, "$NAME.csv"))
