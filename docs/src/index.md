```@meta
CurrentModule = EllSparse
```

# EllSparse.jl

SELL-C (Sliced ELLpack) sparse matrix storage with CPU and GPU [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels for `y = A*x`, `y = A'*x`, and their in-place `mul!` forms.

It is built for matrices whose rows all hold about the same number of entries, such as stencil, mesh, and nearest-neighbor operators. In that regime, CSC pays for indirection it doesn't need. ELL-style storage instead turns each row into a fixed-length dense dot product: the work is embarrassingly parallel over rows, with no per-row bookkeeping.

## Quick start

```julia
using EllSparse, SparseArrays, LinearAlgebra

S = sprand(1_000, 1_000, 0.01)
A = sell(S)                      # SELL-1: row-contiguous, the CPU layout
x = rand(1_000)

A * x ≈ S * x                    # forward apply
A' * x ≈ S' * x                  # adjoint apply, deterministic and atomics-free
mul!(similar(x), A, x, 2.0, 0.0) # 5-arg mul! with the α/β contract

A32 = reslice(A, Val(32))        # SELL-32: warp-coalesced, the GPU layout
SparseMatrixCSC(A32) == S        # round-trips exactly

# Uniform-width rows straight from k × m value/index matrices (zero-copy)
k, m = 5, 1_000
vals, idx = rand(k, m), rand(Int32(1):Int32(m), k, m)
E = SellMatrix(vals, idx, m)     # parent(E) === vals
```

Moving to a GPU is `Adapt.adapt(CuArray, reslice(A, preferred_slice_height(CUDABackend())))`. `adapt` moves data but never changes layout, so a host-to-device copy can't silently change the adjoint's summation order.

## The layout

`SellMatrix{T, C}` cuts the matrix into horizontal slices of `C` consecutive rows and pads each slice to the width of its longest row:

```
         slots (l) →
row 1  ┌ a  b  c ┐
row 2  └ d  e  ⋅ ┘   slice 1 (C = 2 rows, width w₁ = 3; ⋅ = sentinel padding)
row 3  ┌ f  ⋅    ┐
row 4  └ g  h    ┘   slice 2 (width w₂ = 2)
```

Within slice `s`, the entry for (local row `r`, slot `l`) is stored at

```
q = sliceptr[s] + (l - 1) * C + (r - 1)
```

so for a fixed slot, consecutive rows sit next to each other in memory. The one knob, `C`, decides who reads memory contiguously:

  - **`C = 32` (GPU).** A warp of 32 threads, one per row, reads 32 adjacent words per slot, giving coalesced loads. This is the cuSPARSE SELL layout.
  - **`C = 1` (CPU).** Each row's entries are contiguous, which is what one core's SIMD dot product wants. With a uniform width `k` the storage is bit-identical to a column-major `k × m` matrix, which is why the zero-copy `SellMatrix(vals, colind, n)` constructor exists. `EllMatrix` is the alias for `SellMatrix{T, 1}`.

Getting `C` wrong for the device isn't an error, just slow: on CPU, `C = 32` strides through memory by `C` per slot, and on GPU, `C = 1` makes every thread in a warp touch a different cache line. Use [`preferred_slice_height`](@ref) to choose `C` for a backend, and [`reslice`](@ref) to change it explicitly.

## Design decisions

  - **Slice height is a type parameter.** Kernels specialize on `C`, and changing layout is always an explicit [`reslice`](@ref). [`sell`](@ref) takes a runtime `slice_height` and passes it through a `Val` function barrier.
  - **No classic single-slice ELL.** On CPUs, slot-major striding is cache-hostile and `C = 1` wins. On GPUs, `C = 32` strictly dominates it, which is why cuSPARSE dropped plain ELL. Its global-max-width padding remains available as `pad = :global`.
  - **Structure and values are separate.** Everything except `vals` lives in an immutable [`SellStructure`](@ref). [`with_values`](@ref) makes matrices that share it (`===`), so a family of same-pattern matrices (e.g. the components of a gradient operator) stores one column-index array and one transpose map. [`adapt_family`](@ref) moves such a family to a device without breaking the sharing.
  - **Adjoints are deterministic.** The naive `A' * x` is a scatter, so it needs atomics and its summation order depends on thread scheduling. A precomputed [`TransposeMap`](@ref) turns it into a per-column gather with a fixed order (ascending storage position), so results are bit-identical under any thread count. This matters when the adjoint sits inside an AD pullback. [`reslice`](@ref) remaps the transpose map instead of rebuilding it, so the order survives layout changes too.

## Not yet supported

SELL-C-σ row sorting, COO ingestion (construct via `SparseMatrixCSC`), asynchronous batched multi-column `mul!`, cuSPARSE interop (their SELL uses a `-1` sentinel and 0-based offsets), and a parallel transpose-map build.

## Other implementations

In Julia, as of November 2026, no other registered package provides an ELL or SELL matrix type. The GPU wrappers expose the vendor formats only as raw library bindings:

  - [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) wraps `cusparseCreateSlicedEll` and `cusparseCreateBlockedEll` in `CUDA.CUSPARSE`, but its matrix types are CSC, CSR, BSR, and COO only.
  - [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) wraps rocSPARSE's ELL and SELL descriptors (`rocsparse_create_ell_descr`, `rocsparse_create_sell_descr`) and `ellmv`, likewise without a matrix type.
  - [jlust](https://github.com/klamike/jlust), an experimental, unregistered sparse-tensor package, describes SELL and Blocked-ELL as level formats targeting cuSPARSE.

Outside Julia:

  - [NVIDIA cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html#sliced-ellpack-sell): Sliced ELLPACK (SELL) for SpMV and [Blocked-ELL](https://docs.nvidia.com/cuda/cusparse/index.html#blocked-ellpack-blocked-ell) for SpMM.
  - [AMD rocSPARSE](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/conceptual/storage-formats-sparse.html): ELL, Blocked-ELL, and SELL.
  - [Ginkgo](https://github.com/ginkgo-project/ginkgo) (C++; CPU, CUDA, HIP, SYCL): [`Ell`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1matrix_1_1Ell.html) and [`Sellp`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1matrix_1_1Sellp.html).
  - [PETSc](https://petsc.org/release/manualpages/Mat/MATSELL/): `MATSELL`, a SIMD-vectorized sliced ELL for CPUs.
  - [MAGMA](https://icl.utk.edu/magma/): SELL-P, sliced ELL with rows padded so several GPU threads can share one row.
  - [Kreutzer et al. (2014)](https://arxiv.org/abs/1307.6209) introduced SELL-C-σ, the sorted generalization of the layout used here.
