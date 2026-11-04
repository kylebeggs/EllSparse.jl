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
