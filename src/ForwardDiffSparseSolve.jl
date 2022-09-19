module ForwardDiffSparseSolve

using ForwardDiff, LinearAlgebra, SparseArrays

using ForwardDiff: value, partials, Dual, Partials
using SparseArrays: getcolptr, rowvals

import LinearAlgebra: ldiv!, \

function fillpartials!(p, dualarray, j)
    size(p) == size(dualarray) || throw(DimensionMismatch())
    @inbounds @simd for i in eachindex(p, dualarray)
        p[i] = partials(dualarray[i], j)
    end
end

function fillvalues!(v, dualarray)
    size(v) == size(dualarray) || throw(DimensionMismatch())
    @inbounds @simd for i in eachindex(v, dualarray)
        v[i] = value(dualarray[i])
    end
end


struct DualldivTmp{V,Ti}
    # for partials
    A::SparseMatrixCSC{V,Ti}
    b::Vector{V}
    
    # for ouptut
    Yreal::Vector{V}
    Ypart::Matrix{V}
    
    function DualldivTmp(A::SparseMatrixCSC{T,Ti}, b::Vector{T}) where {T<:Dual,Ti}
        V = ForwardDiff.valtype(T)
        K = ForwardDiff.npartials(T)
        
        n = LinearAlgebra.checksquare(A)
        length(b) == n || throw(DimensionMismatch())

        colptr = getcolptr(A)
        rows = rowvals(A)
        
        Yreal = Vector{V}(undef, n)

        v = Vector{V}(undef, nnz(A))
        Atmp  = SparseMatrixCSC(n, n, colptr, rows, v)

        Btmp = similar(Yreal)
        Ypartial = Matrix{V}(undef, n, K)

        return new{V, Ti}(Atmp, Btmp, Yreal, Ypartial)
    end
end

function replaceNaN!(x::AbstractArray{T}) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(x)
        if isnan(x[i])
            x[i] = zero(T)
        end
    end
    return x
end


"""
    ldiv!(Y, A::SparseMatrixCSC{<:Dual}, b::AbstractVector{<:Dual}, tmp::DualldivTmp; f=factorize, replaceNaN=true)

Solve the linear system `A * Y = b` for `Y` using the factorization `f(A)`
for a sparse, dual-valued `A`. 

`tmp` is a `DualldivTmp` object that is used to store temporary data.
"""
function ldiv!(Y::AbstractVector{D}, A::SparseMatrixCSC{D}, b::AbstractVector{D}, tmp::DualldivTmp; f::Function=factorize, replaceNaN=true) where {D<:Dual}
    n = LinearAlgebra.checksquare(A)
    n == length(b) || throw(DimensionMismatch())
    n == length(Y) || throw(DimensionMismatch())
    
    # type signatures
    T = ForwardDiff.tagtype(D)
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    
    # Y = A⁻¹b = Ar \ br
    fillvalues!(nonzeros(tmp.A), nonzeros(A))
    fillvalues!(tmp.b, b)
    Af = f(tmp.A)
    ldiv!(tmp.Yreal, Af, tmp.b)  # real part

    # ∂ⱼ = Ar \ (bpⱼ - Apⱼ*A⁻¹b )
    for j in 1:N
        fillpartials!(nonzeros(tmp.A), nonzeros(A), j)
        Ypj = view(tmp.Ypart, :, j)
        fillpartials!(Ypj, b, j)
        mul!(Ypj, tmp.A, tmp.Yreal, -1, true)
        ldiv!(Af, Ypj)
        if replaceNaN
            replaceNaN!(Ypj)
        end
    end

    # https://stackoverflow.com/questions/68543737/convert-a-tuple-vector-into-a-matrix-in-julia
    @inbounds @simd for i in eachindex(Y)
        val = tmp.Yreal[i]
        tup = NTuple{N,V}(view(tmp.Ypart, i, :))
        part = Partials(tup)
        Y[i] = Dual{T,V,N}(val, part)
    end
    
    return Y
end


function \(A::SparseMatrixCSC{D}, b::AbstractVector{D}; replaceNaN=true) where {D<:Dual}
    Y = similar(b)
    tmp = DualldivTmp(A, b)
    ldiv!(Y, A, b, tmp; f=factorize, replaceNaN=replaceNaN)
    return Y
end


end # module
