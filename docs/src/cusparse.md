# cuSPARSE

NVIDIA's cuSPARSE has its own Sliced-ELL (SELL) format, and it is the same layout as `SellMatrix{T, 32}`: rows cut into slices of 32, each slice padded to its longest row and stored slot-major. Because the bytes already line up, a device `SellMatrix` can be handed to cuSPARSE almost for free. The values and slice offsets are shared as-is, and only the column indices are copied, because the two libraries mark padding differently (`0` here, `-1` in cuSPARSE). This page shows the handoff through CUDA.jl's raw bindings, then compares the two libraries' kernels on the same matrices.

## Calling cuSPARSE SELL from Julia

CUDA.jl has no SELL matrix type, so the descriptor is built with `cusparseCreateSlicedEll` directly:

```julia
using EllSparse, Adapt, CUDA, LinearAlgebra, SparseArrays
using CUDA.CUSPARSE
using CUDA.CUSPARSE: cusparseSpMatDescr_t, cusparseCreateSlicedEll, cusparseDestroySpMat,
    cusparseSpMV_bufferSize, cusparseSpMV, CuDenseVectorDescriptor, handle

# A cuSPARSE SELL descriptor over a device SellMatrix{T, C}, plus the buffers it points into
mutable struct CuSell{T, V, I}
    handle::cusparseSpMatDescr_t
    vals::V
    colind::I
    sliceptr::I
end
Base.unsafe_convert(::Type{cusparseSpMatDescr_t}, M::CuSell) = M.handle

devptr(a) = Ptr{Cvoid}(UInt(pointer(a)))

function CuSell(A::SellMatrix{T, C}) where {T, C}
    st = structure(A)
    colind = map(c -> iszero(c) ? -one(c) : c, st.colind)   # padding sentinel: 0 → -1
    vals, sliceptr = parent(A), st.sliceptr                   # shared, not copied
    ref = Ref{cusparseSpMatDescr_t}()
    GC.@preserve vals colind sliceptr cusparseCreateSlicedEll(
        ref, size(A)..., count(!iszero, st.colind), length(vals), C,
        devptr(sliceptr), devptr(colind), devptr(vals),
        eltype(sliceptr), eltype(colind), 'O', T,              # 'O': one-based
    )
    M = CuSell{T, typeof(vals), typeof(colind)}(ref[], vals, colind, sliceptr)
    return finalizer(M -> cusparseDestroySpMat(M.handle), M)
end

# y ← op(A) x with cuSPARSE, op = 'N' or 'T'
function cusparse_mul!(y, M::CuSell, x; op = 'N')
    T = eltype(y)
    α, β = Ref{T}(one(T)), Ref{T}(zero(T))
    dx, dy = CuDenseVectorDescriptor(x), CuDenseVectorDescriptor(y)
    alg = CUSPARSE.CUSPARSE_SPMV_SELL_ALG1
    sz = Ref{Csize_t}()
    cusparseSpMV_bufferSize(handle(), op, α, M, dx, β, dy, T, alg, sz)
    buf = CuVector{UInt8}(undef, max(sz[], 1))
    cusparseSpMV(handle(), op, α, M, dx, β, dy, T, alg, buf)
    return y
end

S = sprand(10_000, 10_000, 0.001) + I
A = adapt(CuArray, sell(S; slice_height = 32))
M = CuSell(A)
x = CUDA.rand(Float64, 10_000)
cusparse_mul!(similar(x), M, x) ≈ A * x   # true
```

Three details are easy to get wrong:

  - **The index base covers the slice offsets too.** With `'O'`, cuSPARSE treats both `sellColInd` and `sellSliceOffsets` as one-based, so EllSparse's `sliceptr` can be passed unchanged. Mixing a one-based base with zero-based offsets ends in `ERROR_ILLEGAL_ADDRESS`, not an exception.
  - **CUDA.jl's binding takes host pointers.** The generated `cusparseCreateSlicedEll` declares its buffers as `Ptr{Cvoid}` instead of `CuPtr{Cvoid}` (the CSR and Blocked-ELL bindings use `CuPtr`), so a `CuArray` won't convert on its own. `devptr` reinterprets the device address. That is safe only while the arrays stay alive, which is why `CuSell` holds them.
  - **The descriptor holds raw pointers, not arrays.** Whatever owns `vals`, `colind`, and `sliceptr` must outlive every call that uses the descriptor. Passing the `CuSell` object itself to `cusparseSpMV`, through `unsafe_convert`, roots it for the duration of the call.

In real use, allocate the workspace once and reuse it rather than calling `cusparseSpMV_bufferSize` on every product. [`benchmark/cusparse.jl`](https://github.com/kylebeggs/EllSparse.jl/blob/main/benchmark/cusparse.jl) does this.

## Benchmarks

Each row below is one matrix, run three ways on the same device: EllSparse's own KernelAbstractions kernels on a `SellMatrix{Float64, 32}`, cuSPARSE SELL reading that same matrix through `CuSell`, and cuSPARSE CSR as the usual baseline. All products are Float64 and every result is checked against the host `SparseMatrixCSC` product before timing. Times are minimum μs per product. cuSPARSE descriptors, workspaces, and `cusparseSpMV_preprocess` are set up once outside the timed region.

Measured on an NVIDIA RTX 4000 Ada Generation (360 GB/s), cuSPARSE 12.8.6, CUDA.jl 6.5.1, Julia 1.13.3.

### Forward product `A * x`

| Matrix | nnz/row | SELL padding | EllSparse | cuSPARSE SELL | cuSPARSE CSR |
|:--|--:|--:|--:|--:|--:|
| 2D Laplacian, 5-pt (1000²) | 5.0 | 0.0% | 241.6 | 186.5 | 201.9 |
| 3D 27-pt stencil (64³) | 26.2 | 1.0% | 276.6 | 255.8 | 262.8 |
| kNN stencil, k = 32 (1M) | 30.1 | 5.8% | 1213.5 | 1153.3 | 1104.1 |
| random columns, k = 32 (1M) | 32.0 | 0.0% | 1204.9 | 1166.7 | 1236.2 |
| ragged rows, 1–64 (1M) | 30.0 | 46.9% | 1894.8 | 2035.8 | 1099.9 |

Forward SpMV is a streaming problem. Every stored value and index is read exactly once, so the runtime floor is bytes over bandwidth. For the 1M-row matrices that is about 360 MB (12 bytes per entry) at 360 GB/s, or roughly 1 ms, which is where all three implementations land. Once the matrix is large enough to hide launch costs, the layout decides the time, not the kernel author. That is why EllSparse comes within about 5% of cuSPARSE SELL there, and why its gap on the small 2D Laplacian (about 0.2 ms of work) is the largest.

The ragged row is the exception, and it is a layout cost rather than a kernel cost. A slice is padded to its longest row, so when row lengths vary from 1 to 64, nearly half of the stored entries are padding, and both SELL implementations stream those zeros. CSR stores no padding and wins by 1.7–1.9×. Sorting rows by length before slicing (SELL-C-σ) is the standard fix, and EllSparse does not do that yet.

### Adjoint product `A' * x`

| Matrix | EllSparse | cuSPARSE SELL | cuSPARSE CSR |
|:--|--:|--:|--:|
| 2D Laplacian, 5-pt (1000²) | 312.7 | 193.4† | 194.1† |
| 3D 27-pt stencil (64³) | 469.6 | 258.6† | 255.9† |
| kNN stencil, k = 32 (1M) | 3118.0 | 1148.4† | 1104.2† |
| random columns, k = 32 (1M) | 13235.5 | 1164.7† | 1165.6† |
| ragged rows, 1–64 (1M) | 4439.0 | 1821.5† | 1093.2† |

† Not bitwise-repeatable: repeated runs on identical input gave different last-bit results. On the kNN matrix, 379k of the 1M output entries differed between two consecutive runs.

The adjoint is where the two libraries make different choices. Computing `Aᵀx` from a row-major layout is a scatter: each row adds its contribution into many output entries, and threads collide on the same outputs. cuSPARSE resolves the collisions with atomic adds. That runs as fast as the forward product, but the order of floating-point additions depends on thread scheduling, so the result changes in the last bits from run to run. This held for both the SELL and CSR algorithms, including `CUSPARSE_SPMV_CSR_ALG2`. The forward products were bitwise-repeatable in all three implementations.

EllSparse turns the scatter into a gather instead. A precomputed [`TransposeMap`](@ref) lists, for each column, where its entries live in storage, so each output is a private sum taken in a fixed order. The price is indirection. Every term reads a storage position from the map, then the value at that position, then the matching `x` entry, and the value reads are no longer contiguous. On local stencils that costs 1.6–2.7× cuSPARSE. With uniformly random columns, where nothing about one column's storage positions predicts the next, it costs 11×.

So the choice depends on what the adjoint is for. If `A' * x` sits inside an AD pullback, an adjoint test, or anything else that needs bit-for-bit reproducibility, the gather is the point. If it is a throughput-bound inner loop where last-bit noise is acceptable, cuSPARSE's atomic scatter, reachable through `CuSell`, is several times faster.

### Reproducing

```sh
julia --project=benchmark --startup-file=no -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=benchmark --startup-file=no benchmark/cusparse.jl
```

The script prints both tables in this format.
