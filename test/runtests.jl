using SafeTestsets

# One @safetestset per file: each runs in its own module, which the files' shared
# JLArrays `synchronize` shim (hasmethod-guarded) relies on.
@safetestset "Construction" begin
    include("construction.jl")
end

@safetestset "Forward Apply" begin
    include("mul.jl")
end

@safetestset "Adjoint Apply" begin
    include("adjoint.jl")
end

@safetestset "Structure Sharing" begin
    include("structure_sharing.jl")
end

@safetestset "Conversions and Algebra" begin
    include("conversions_algebra.jl")
end

@safetestset "Reslice" begin
    include("reslice.jl")
end

@safetestset "Code linting (JET.jl)" begin
    using EllSparse, JET
    JET.test_package(EllSparse; target_modules = (EllSparse,))
end
