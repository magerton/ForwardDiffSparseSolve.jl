# using Revise
using ForwardDiffSparseSolve
using Test
using LinearAlgebra, SparseArrays, ForwardDiff

using ForwardDiff: Dual

const fdmt = ForwardDiffSparseSolve

@testset "ForwardDiffSparseSolve.jl" begin

    xdual = [Dual(1.0, 1.0, 0.0), Dual(2.0, 0.0, 1.0)]
    b = rand(2).*xdual

    # check dense M
    Mpattern = [1.0 1; 0 1] # rand(2,2)
    Mdense = Mpattern.*xdual
    Ydense = Mdense\b

    # check sparse M
    Msp = sparse(Mpattern).*xdual

    # fixed in ForwardDiff#481?
    @test_broken Dual(1, 1, 0) != Dual(1,NaN, 0)
    @test_broken Ydense != \(Msp, b; replaceNaN=false)
    
    @test        Ydense == \(Msp, b; replaceNaN=true)
    @test Msp*(Msp\b) ≈ b

    tmp = fdmt.DualldivTmp(Msp, b)
    Y = similar(b)

    ldiv!(Y, Msp, b, tmp; replaceNaN=false)
    @test_broken Ydense != Y  # fixed in ForwardDiff#481?
    
    ldiv!(Y, Msp, b, tmp; replaceNaN=true)
    @test Ydense == Y


    AA = [Dual(rand(3)...) for i in 1:2, j in 1:2]
    bb = [Dual(rand(3)...) for i in 1:2]
    AAsp = sparse(AA)
    AAreal = ForwardDiff.value.(AA)
    bbreal=  ForwardDiff.value.(bb)
    tmp2 = fdmt.DualldivTmp(AAsp, bb)
    Y2 = similar(bb)

    @test AA\bb ≈ \(AAsp, bb; replaceNaN=false)
    @test AA\bb ≈ \(AAsp, bb; replaceNaN=true)
    @test bb ≈ AAsp*\(AAsp, bb; replaceNaN=true)
    @test bb ≈ AA*(AA\bb)

    ldiv!(Y2, AAsp, bb, tmp2; replaceNaN=false, f=lu)
    @test AA\bb ≈ Y2

end

