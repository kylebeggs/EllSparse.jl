# EllSparse

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kylebeggs.github.io/EllSparse.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kylebeggs.github.io/EllSparse.jl/dev/)
[![Build Status](https://github.com/kylebeggs/EllSparse.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kylebeggs/EllSparse.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kylebeggs/EllSparse.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kylebeggs/EllSparse.jl)

SELL-C (Sliced ELLpack) sparse matrix storage with CPU and GPU [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels. Built for matrices with roughly uniform row lengths (stencil, mesh, and nearest-neighbor operators), where each row becomes a fixed-length dense dot product. The slice height `C` is a type parameter: `C = 1` is row-contiguous for CPU SIMD, `C = 32` is warp-coalesced for GPUs. Adjoint products use a precomputed transpose map, so they are deterministic and need no atomics.

Extracted from [RadialBasisFunctions.jl](https://github.com/JuliaMeshless/RadialBasisFunctions.jl).

```julia
using EllSparse, SparseArrays, LinearAlgebra

S = sprand(1_000, 1_000, 0.01)
A = sell(S)                 # SELL-1, the CPU layout
x = rand(1_000)
A * x ≈ S * x
A' * x ≈ S' * x             # deterministic, atomics-free adjoint

A32 = reslice(A, Val(32))   # SELL-32, the GPU layout; move with Adapt.adapt
```
