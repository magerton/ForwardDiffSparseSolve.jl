module ForwardDiffMatrixTools

using ForwardDiff, LinearAlgebra, SparseArrays # , SuiteSparse

using ForwardDiff: value, partials, Dual, Partials
using SparseArrays: getcolptr, rowvals

import Base.\
import LinearAlgebra: factorize
export \, factorize

# low mem-usage partials & values
for f in (:partials, :value,)
    @eval begin
        import ForwardDiff: $f
        """
            $($f)(M::SparseMatrixCSC{<:ForwardDiff.Dual, <:Int})
        Returns a sparse matrix with the `$($f)` of the `Dual`s in `M`.

        NOTE: does NOT allocate new colptr or rowval arrays.        
        """
        function $f(M::SparseMatrixCSC{<:Dual,<:Int})
            SparseMatrixCSC(size(M)..., getcolptr(M), rowvals(M), $f.(nonzeros(M)))
        end
        export $f
    end
end

"return SparseMatrixCSC of jth partials (∂M/∂xⱼ)"
function partials(M::SparseMatrixCSC{<:Dual,<:Int},j)
    m,n = size(M)
    colptr = getcolptr(M)
    rows = rowvals(M)
    v = partials.(nonzeros(M), j)
    return SparseMatrixCSC(m, n, colptr, rows, v)
end


"""
    FDFactor

Container type to work efficiently with backslash on dual-valued sparse matrices.

`factorize(M)` will create an instance containing
- `factor = factorize(value(M))` — the factors of the real part
- `partials = partials(M)` — the dual part
for a dual-valued matrix `M`.

This is because only the factors of the real part are needed when solving 
a linear system of the type ``M x = b`` for a dual-valued matrix ``M = A + \\varepsilon B``.
In fact, the inverse of ``M`` is given by
``M^{-1} = (I - \\varepsilon A^{-1} B) A^{-1}``.
"""
struct FDFactor{T, F<:Factorization, P<:AbstractMatrix}
    factor::F # factors of the real part
    partials::P  # partials(M)
end
export FDFactor

function FDFactor(T, fac::F, part::P) where {F,P}
    return FDFactor{T,F,P}(fac, part)
end


export factor
factor(x::FDFactor) = x.factor
partials(x::FDFactor) = x.partials

function partials(x::FDFactor, j)
    p = partials(x)
    v = getindex.(nonzeros(p), j)
    return SparseMatrixCSC(size(p)..., getcolptr(p), rowvals(p), v)
end



import ForwardDiff: tagtype, npartials, valtype
tagtype(::FDFactor{T}) where {T} = T
npartials(x::FDFactor) = npartials(eltype(partials(x)))






# Factorization functions
for f in (:lu, :qr, :cholesky, :factorize)
    @eval begin
        import LinearAlgebra: $f
        
        
        # sparse
        """
            $($f)(M::SparseMatrixCSC{<:Dual,<:Int})

        Invokes `$($f)` on just the real part of `M` and stores 
        it along with the partials into a `FDSparseFactor` object.
        """
        function $f(M::SparseMatrixCSC{<:Dual{T},<:Int}) where {T}
            return FDFactor(T, $f(value(M)), partials(M))
        end
        
        
        
        
        # dense
        """
            $($f)(M::Array{<:Dual,2})

        Invokes `$($f)` on just the real part of `M` and stores 
        it along with the partials into a `FDSparseFactor` object.
        """
        function $f(M::Matrix{<:Dual{T}})  where {T}
            return FDFactor(T, $f(value.(M)), partials.(M))
        end
        
        
        export $f
    end
end

# https://stackoverflow.com/questions/68543737/convert-a-tuple-vector-into-a-matrix-in-julia




"""
    \\(M::FDFactor, y::AbstractVecor{<:AbstractFloat})

Backsubstitution for `FDFactor`.
See `DualFactors` for details.

    M-1 = (I - ε A-1 B) A-1
"""
function \(A::FDFactor{T}, b::AbstractVector{<:AbstractFloat}) where {T}
    n = npartials(A)
    m = length(b)
    Ar = factor(A)
    V = valtype(eltype(partials(A)))
    N = npartials(A)
    
    Ar⁻¹y = Ar \ b  # outreal
    outreal = Ar⁻¹y

    partialmat = Matrix{V}(undef, m, n)
    tmp = Vector{V}(undef, m)
    for j in 1:n
        Apj = partials(A,j)
        mul!(tmp, Apj, Ar⁻¹y, -1, false)
        partialj = view(partialmat, :,j)
        ldiv!(partialj, Ar, tmp)
    end

    outvec = Vector{Dual{T,V,N}}(undef, m)
    for j in eachindex(outvec)
        val = outreal[j]
        part = Partials(tuple(partialmat[j,:]...))
        outvec[j] = Dual{T,V,N}(val, part)
    end
    return outvec
end

# """
#     \\(M::DualFactors, y::AbstractVecOrMat{Dual128})

# Backsubstitution for `DualFactors`.
# See `DualFactors` for details.
# """
# function \(M::DualFactors, y::AbstractVecOrMat{Dual128})
#     a, b = realpart.(y), dualpart.(y)
#     A, B = M.Af, M.B
#     A⁻¹a = A \ a
#     return A⁻¹a + ε * (A \ (b - B * A⁻¹a))
# end

# """
#     \\(Af::Factorization{Float64}, y::AbstractVecOrMat{Dual128})

# Backsubstitution for Dual-valued RHS.
# """
# function \(Af::Factorization{Float64}, y::AbstractVecOrMat{Dual128})
#     return (Af \ realpart.(y)) + ε * (Af \ dualpart.(y))
# end

# """
#     \\(M::AbstractArray{Dual128,2}, y::AbstractVecOrMat)

# Backslash (factorization and backsubstitution) for Dual-valued matrix `M`.
# """
# \(M::SparseMatrixCSC{<:Dual,<:Int}, y::AbstractVecOrMat) = factorize(M) \ y
# \(M::Array{<:Dual,2}, y::AbstractVecOrMat) = factorize(M) \ y


end # module
