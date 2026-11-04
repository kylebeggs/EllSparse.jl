#=
GPU sweep: EllSparse vs the GPU sparse-matrix options a Julia user can install today —
CUDA.jl's cuSPARSE-backed CSR, CSC, and COO types, and GenericSparseArrays.jl's
KernelAbstractions CSR and CSC on CuArrays — over matrix size × family × operation.
Not run in CI — this needs real CUDA hardware.

Run from the repo root on a CUDA machine:

    julia --project=benchmark --startup-file=no -e 'using Pkg; Pkg.instantiate()'   # once
    julia --project=benchmark --startup-file=no benchmark/sweep_gpu.jl

Every format is called through plain `mul!`, the path a user takes (CUDA.jl manages
cuSPARSE descriptors and workspaces itself). Results go to benchmark/results/gpu.csv;
plot with benchmark/plot.jl.
=#

using Adapt
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCOO, CuSparseMatrixCSC, CuSparseMatrixCSR
using EllSparse
using GenericSparseArrays

CUDA.functional() || error("CUDA is not functional on this machine")

include("sweep_common.jl")

const MAX_NNZ = parse(Float64, get(ENV, "BENCH_MAX_NNZ", "1.2e8"))

const DEVICE = CUDA.name(CUDA.device())
const BASE = (; backend = "gpu", device = DEVICE, threads = 0)
gsync(f) = CUDA.@sync f()

# One device matrix per format; every format supports all three operations through mul!
function formats(S)
    return [
        ("EllSparse", adapt(CuArray, sell(S; slice_height = 32))),
        ("CUDA.jl CSR", CuSparseMatrixCSR(S)),
        ("CUDA.jl CSC", CuSparseMatrixCSC(S)),
        ("CUDA.jl COO", CuSparseMatrixCOO(S)),
        ("GenericSparseArrays CSR", adapt(CuArray, GenericSparseMatrixCSR(S))),
        ("GenericSparseArrays CSC", adapt(CuArray, GenericSparseMatrixCSC(S))),
    ]
end

println("device: $DEVICE   cuSPARSE $(CUSPARSE.version())")
println("formats: ", join(first.(formats(int32(laplacian2d(16)))), ", "), "\n")

let (a, b, c) = (CUDA.zeros(Float64, 2^27), CUDA.rand(Float64, 2^27), CUDA.rand(Float64, 2^27))
    l2 = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    emit_meta("gpu"; device = DEVICE, threads = 0, triad_gbs = triad_gbs(a, b, c; sync = gsync),
        llc_bytes = l2, julia = string(VERSION), cusparse = string(CUSPARSE.version()))
end

fresh = Ref(true)
for (fam, Ntarget) in sweep_cases(MAX_NNZ)
    S = int32(fam.build(Ntarget))
    (m, n) = size(S)
    pad32 = sell_padding(S)
    println("$(fam.label): $m rows, nnz $(nnz(S)), SELL-32 padding $(round(100pad32; digits = 1))%")
    meta = (; BASE..., family = fam.key, label = fam.label, m, nnz = nnz(S), pad32)

    (xh, vh) = (randn(MersenneTwister(3), n), randn(MersenneTwister(4), m))
    (x, v) = (CuArray(xh), CuArray(vh))
    (yref, zref) = (ref_mul(S, xh), ref_adj(S, vh))
    Xhs = map(r -> randn(MersenneTwister(5), n, r), NRHS)
    Xrefs = map(Xh -> reduce(hcat, map(c -> ref_mul(S, Xh[:, c]), axes(Xh, 2))), Xhs)
    (y, z) = (CUDA.zeros(Float64, m), CUDA.zeros(Float64, n))
    rows = []
    record!(r) = (push!(rows, r); report(r))

    foreach(formats(S)) do (name, A)
        meta_f = (; meta..., format = name, group = "gpu")
        fwd() = mul!(y, A, x)
        adj() = mul!(z, A', v)

        gsync(fwd)
        relerr(y, yref) < 1.0e-12 || error("$name A*x mismatch: $(relerr(y, yref))")
        record!(row(; meta_f..., op = "spmv", useful_bytes = useful_bytes(S), time_s = timeit(fwd; sync = gsync)))

        gsync(adj)
        relerr(z, zref) < 1.0e-12 || error("$name A'x mismatch: $(relerr(z, zref))")
        record!(row(; meta_f..., op = "spmv_adj", useful_bytes = useful_bytes(S),
            time_s = timeit(adj; sync = gsync), deterministic = repeatable(() -> gsync(adj), z)))

        foreach(zip(NRHS, Xhs, Xrefs)) do (r, Xh, Xref)
            (X, Y) = (CuArray(Xh), CUDA.zeros(Float64, m, r))
            spmm() = mul!(Y, A, X)
            gsync(spmm)
            relerr(Y, Xref) < 1.0e-12 || error("$name A*X (r = $r) mismatch: $(relerr(Y, Xref))")
            record!(row(; meta_f..., op = "spmm", nrhs = r, useful_bytes = useful_bytes(S, r),
                time_s = timeit(spmm; sync = gsync)))
        end
        GC.gc(true)
        CUDA.reclaim()
    end

    emit!("gpu", rows, fresh)
end
println("\nwrote ", joinpath(RESULTS, "gpu.csv"))
