#=
Test-matrix generators shared by the benchmark scripts. Every generator is seeded, so a
given size always produces the same matrix.
=#

using LinearAlgebra
using Random: MersenneTwister
using SparseArrays

laplacian1d(n) = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))

function laplacian2d(n)
    L, Id = laplacian1d(n), sparse(1.0I, n, n)
    return kron(L, Id) + kron(Id, L)                 # 5-point, n² rows
end

function laplacian3d(n)
    L, Id = laplacian1d(n), sparse(1.0I, n, n)
    return kron(L, kron(Id, Id)) + kron(Id, kron(L, Id)) + kron(Id, kron(Id, L))   # 7-point, n³ rows
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

# ── Size sweep ────────────────────────────────────────────────────────────────────
#
# Each family is built from a target row count N; grid families round N to the nearest
# cube/square side.

const FAMILIES = [
    (key = "lap2d", label = "2D Laplacian, 5-pt", k = 5, build = N -> laplacian2d(round(Int, sqrt(N)))),
    (key = "lap3d", label = "3D Laplacian, 7-pt", k = 7, build = N -> laplacian3d(round(Int, cbrt(N)))),
    (key = "st27", label = "3D 27-pt stencil", k = 27, build = N -> stencil27(round(Int, cbrt(N)))),
    (key = "knn", label = "kNN stencil, k = 32", k = 32, build = N -> knn_stencil(N, 32)),
    (key = "ragged", label = "ragged rows, 1–64", k = 32, build = N -> ragged(N, 32)),
]

# Log-spaced target row counts, 10³ … 10⁷
const SIZES = map(e -> round(Int, 10.0^e), 3:0.5:7)

# The (family, N) pairs whose estimated nnz fits under `max_nnz`, smallest first.
sweep_cases(max_nnz) = [(f, N) for f in FAMILIES for N in SIZES if f.k * N ≤ max_nnz]

# Int32 indices for every format, so all of them stream the same 12 bytes per entry.
int32(S::SparseMatrixCSC) = SparseMatrixCSC{Float64, Int32}(S)
