using EllSparse
using Test
using JET

@testset "EllSparse.jl" begin
    @testset "Code linting (JET.jl)" begin
        JET.test_package(EllSparse; target_defined_modules = true)
    end
    # Write your tests here.
end
