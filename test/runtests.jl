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

    FD.tagtype(luSF)
    FD.valtype(luSF)
    FD.npartials(luSF)
    
    v = Vector{Float64}(undef, length(nonzeros(Msp)))
    @code_warntype FDMT.partials!(v, luSF, 1)

    @test factor(luSF).L == lubase.L
    @test factor(luSF).U == lubase.U
    @test factor(luSF).p == lubase.p

    fill(first(xdual), 10)

    FD.npartials(luSF)

    FD.npartials(eltype(partials(Msp)))

    SparseArrays.nonzeros(Msp)

    b = rand(2)
    luSF \ b
    @test luSF \ b ≈ Mdense\ b
    
    tmp = FDMT.FDFactorTmp(Msp)
    Y = Vector{eltype(Msp)}(undef, size(Msp, 1))

    @test FDMT.myldiv!(Y, tmp, luSF, b) == luSF\b
    @btime FDMT.myldiv!(Y, tmp, luSF, b)
    @btime luSF\b

    @code_warntype FDMT.myldiv!(Y, tmp, luSF, b)

    FDMT.jnk(luSF)




# end
