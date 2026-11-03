# EllSparse

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kylebeggs.github.io/EllSparse.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kylebeggs.github.io/EllSparse.jl/dev/)
[![Build Status](https://github.com/kylebeggs/EllSparse.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kylebeggs/EllSparse.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kylebeggs/EllSparse.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kylebeggs/EllSparse.jl)

SELL-C (Sliced ELLpack) sparse matrix storage with CPU and GPU [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels. Built for matrices with roughly uniform row lengths (stencil, mesh, and nearest-neighbor operators), where each row becomes a fixed-length dense dot product. The slice height `C` is a type parameter: `C = 1` is row-contiguous for CPU SIMD, `C = 32` is warp-coalesced for GPUs. Adjoint products use a precomputed transpose map, so they are deterministic and need no atomics.

```julia
using EllSparse, SparseArrays, LinearAlgebra

S = sprand(1_000, 1_000, 0.01)
A = sell(S)                 # SELL-1, the CPU layout
x = rand(1_000)
A * x ≈ S * x
A' * x ≈ S' * x             # deterministic, atomics-free adjoint

A32 = reslice(A, Val(32))   # SELL-32, the GPU layout; move with Adapt.adapt
```

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
