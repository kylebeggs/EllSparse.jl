#=
GPU format sweep: EllSparse SELL-32 vs cuSPARSE SELL (reading EllSparse's own buffers)
and cuSPARSE CSR, over matrix size × family × operation. Not run in CI — this needs
real CUDA hardware.

Run from the repo root on a CUDA machine:

    julia --project=benchmark --startup-file=no -e 'using Pkg; Pkg.instantiate()'   # once
    julia --project=benchmark --startup-file=no benchmark/sweep_gpu.jl

SpMV timings reuse a preallocated cuSPARSE workspace and descriptor (kernels, not setup),
as benchmark/cusparse.jl does. SpMM calls cuSPARSE CSR through CUDA.jl's own `mul!`,
the path a user gets; cuSPARSE SELL has no SpMM, so it appears in SpMV only. Results go to benchmark/results/gpu.csv; plot with benchmark/plot.jl.
=#

using Adapt
using CUDA

CUDA.functional() || error("CUDA is not functional on this machine")

include("cusell.jl")
include("sweep_common.jl")

const MAX_NNZ = parse(Float64, get(ENV, "BENCH_MAX_NNZ", "1.2e8"))
const SELL_ALG = CUSPARSE.CUSPARSE_SPMV_SELL_ALG1
const CSR_ALG = CUSPARSE.CUSPARSE_SPMV_CSR_ALG1

# ── Sweep ────────────────────────────────────────────────────────────────────────

const DEVICE = CUDA.name(CUDA.device())
const BASE = (; backend = "gpu", device = DEVICE, threads = 0)
gsync(f) = CUDA.@sync f()

println("device: $DEVICE   cuSPARSE $(CUSPARSE.version())")
println("formats: EllSparse SELL-32, cuSPARSE SELL, cuSPARSE CSR\n")

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

    A = adapt(CuArray, sell(S; slice_height = 32))
    Acsr = CuSparseMatrixCSR(S)
    sellA = CuSell(A)
    csrD = CuSparseMatrixDescriptor(Acsr, 'O')

    (xh, vh) = (randn(MersenneTwister(3), n), randn(MersenneTwister(4), m))
    (x, v) = (CuArray(xh), CuArray(vh))
    (yref, zref) = (ref_mul(S, xh), ref_adj(S, vh))
    rows = []
    record!(r) = (push!(rows, r); report(r))

    # SpMV forward and adjoint
    y = CUDA.zeros(Float64, m)
    z = CUDA.zeros(Float64, n)
    spmv = [
        ("EllSparse SELL-32", () -> mul!(y, A, x), () -> mul!(z, A', v)),
        ("cuSPARSE SELL", spmv_plan(sellA, 'N', x, y, SELL_ALG), spmv_plan(sellA, 'T', v, z, SELL_ALG)),
        ("cuSPARSE CSR", spmv_plan(csrD, 'N', x, y, CSR_ALG), spmv_plan(csrD, 'T', v, z, CSR_ALG)),
    ]
    foreach(spmv) do (name, fwd, adj)
        gsync(fwd)
        relerr(y, yref) < 1.0e-12 || error("$name A*x mismatch: $(relerr(y, yref))")
        record!(row(; meta..., op = "spmv", format = name, useful_bytes = useful_bytes(S),
            time_s = timeit(fwd; sync = gsync)))
        gsync(adj)
        relerr(z, zref) < 1.0e-12 || error("$name A'x mismatch: $(relerr(z, zref))")
        record!(row(; meta..., op = "spmv_adj", format = name, useful_bytes = useful_bytes(S),
            time_s = timeit(adj; sync = gsync), deterministic = repeatable(() -> gsync(adj), z)))
    end

    # SpMM: a dense block of right-hand sides
    foreach(NRHS) do r
        Xh = randn(MersenneTwister(5), n, r)
        Xref = reduce(hcat, map(c -> ref_mul(S, Xh[:, c]), 1:r))
        X = CuArray(Xh)
        Y = CUDA.zeros(Float64, m, r)
        foreach((("EllSparse SELL-32", A), ("cuSPARSE CSR", Acsr))) do (name, M)
            gsync(() -> mul!(Y, M, X))
            relerr(Y, Xref) < 1.0e-12 || error("$name A*X (r = $r) mismatch: $(relerr(Y, Xref))")
            record!(row(; meta..., op = "spmm", format = name, nrhs = r, useful_bytes = useful_bytes(S, r),
                time_s = timeit(() -> mul!(Y, M, X); sync = gsync)))
        end
    end

    emit!("gpu", rows, fresh)
    (A, Acsr, sellA, csrD) = ntuple(_ -> nothing, 4)
    GC.gc(true)
    CUDA.reclaim()
end
println("\nwrote ", joinpath(RESULTS, "gpu.csv"))
