using Revise
using ForwardDiffMatrixTools
using Test
using ForwardDiff
using LinearAlgebra, SparseArrays
using BenchmarkTools

using ForwardDiff: value, partials, Dual, Partials, tagtype, GradientConfig, seed!
using SparseArrays: getcolptr, rowvals

const fdmt = ForwardDiffMatrixTools

# @testset "ForwardDiffMatrixTools.jl" begin

    # generate dual 
    theta = [2.0,2.0]
    f(x) = sin(x)
    cfg = GradientConfig(f, theta)
    tagtype(cfg)
    xdual = cfg.duals
    seed!(xdual, theta, cfg.seeds)

    b = rand(2).*xdual

    # check dense M
    Mpattern = [1.0 1; 0 1] # rand(2,2)
    Mdense = Mpattern.*xdual

    # check sparse M
    Msp = sparse(Mpattern).*xdual

    @test Mdense\b == Msp\b
    @test Msp*(Msp\b) ≈ b

    
    Msp*(Msp\b)

    tmp = fdmt.DualldivTmp(Msp, b)
    Y = similar(b)

    ldiv!(Y, Msp, b, tmp; replaceNaN=false)
    ldiv!(Y, Msp, b, tmp; replaceNaN=true)
    Msp\b

    Msp\b == ldiv!(Y, Msp, b, tmp)
    
    Mdense\b

    @code_lowered fdmt.replaceNaN!(tmp.Yreal)

# end
