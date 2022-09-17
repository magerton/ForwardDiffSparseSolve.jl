using Revise
using ForwardDiffMatrixTools
using Test
using ForwardDiff
using LinearAlgebra, SparseArrays
using BenchmarkTools

using ForwardDiff: value, partials, Dual, Partials, tagtype
using SparseArrays: getcolptr, rowvals

const FD = ForwardDiff
const FDMT = ForwardDiffMatrixTools

# @testset "ForwardDiffMatrixTools.jl" begin

    # generate dual 
    theta = [2.0,2.0]
    f(x) = sin(x)
    cfg = FD.GradientConfig(f, theta)
    tagtype(cfg)
    xdual = cfg.duals
    FD.seed!(xdual, theta, cfg.seeds)

    # check dense M
    Mpattern = [1 1; 0 1]
    Mdense = Mpattern.*xdual

    # check sparse M
    Msp = sparse(Mpattern).*xdual
    @test partials(Msp) == partials.(Msp)
    @test rowvals(partials(Msp)) === rowvals(Msp)
    @test getcolptr(partials(Msp)) === getcolptr(Msp)

    
    luSF = lu(Msp)
    lubase = lu(value(Msp))

    tagtype(luSF)
    
    @test factor(luSF).L == lubase.L
    @test factor(luSF).U == lubase.U
    @test factor(luSF).p == lubase.p

    fill(first(xdual), 10)

    FD.npartials(luSF)

    FD.npartials(eltype(partials(Msp)))

    b = rand(2)
    @test luSF \ b ≈ Mdense\ b
    




# end
