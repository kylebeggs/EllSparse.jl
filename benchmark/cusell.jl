#=
cuSPARSE Sliced-ELL (SELL) view of a device SellMatrix{T, 32}, shared by the GPU
benchmark scripts. See docs/src/cusparse.md for the walkthrough.

Both libraries store a slice column-major (slot-major), so the buffers line up. With a
one-based index base cuSPARSE reads EllSparse's `sliceptr` and values as-is; only the
padding sentinel differs (EllSparse 0, cuSPARSE -1), so `colind` is the one copy.
=#

using CUDA
using CUDA.CUSPARSE
using CUDA.CUSPARSE: cusparseSpMatDescr_t, cusparseCreateSlicedEll, cusparseDestroySpMat,
    cusparseSpMV_bufferSize, cusparseSpMV_preprocess, cusparseSpMV, CuSparseMatrixDescriptor,
    CuDenseVectorDescriptor, handle
using EllSparse
using LinearAlgebra

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
