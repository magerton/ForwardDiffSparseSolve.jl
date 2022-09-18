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
npartials(::FDFactor{T,F,<:AbstractMatrix{Partials{N,V}}}) where {T,F,N,V} = N
valtype(::FDFactor{T,F,<:AbstractMatrix{Partials{N,V}}}) where {T,F,N,V} = V






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


struct FDFactorTmp{V,Ti}
    partials_in::SparseMatrixCSC{V,Ti}
    partials_out::Matrix{V}
    tmp_real::Vector{V}
    
    function FDFactorTmp(partials_in::SparseMatrixCSC{V,Ti}, partials_out, tmp_real) where {V,Ti}
        m,n = size(partials_in)
        length(tmp_real) == m || throw(DimensionMismatch("tmp_real must be of length $m"))
        size(partials_out, 1) == m || throw(DimensionMismatch("partials_out must have $m rows"))
        return new{V,Ti}(partials_in, partials_out, tmp_real)
    end
end

Ar⁻¹y(x::FDFactorTmp) = x.tmp_real

function FDFactorTmp(M::SparseMatrixCSC{<:Dual{T,V,N}}) where {T,V,N}
    m,n = size(M)
    colptr = getcolptr(M)
    rows = rowvals(M)
    
    partials_in_values = Vector{V}(undef, nnz(M))
    partials_in  = SparseMatrixCSC(m, n, colptr, rows, partials_in_values)
    
    partials_out = Matrix{V}(undef, m, N)
    tmp_real     = Vector{V}(undef, m)

    return FDFactorTmp(partials_in, partials_out, tmp_real)
end

function fill_partials!(x::FDFactorTmp, M::FDFactor, j)
    nzM = nonzeros(partials(M))
    nzx = nonzeros(x.partials_in)
    length(nzM) == length(nzx) || throw(DimensionMismatch("partials_in and partials(M) must have the same number of nonzeros"))
    
    @inbounds @simd for i in eachindex(nzM, nzx)
        nzx[i] = getindex(nzM[i], j)
    end
end


"""
    myldiv!(Y, A::FDFactor, b::AbstractVecor{<:AbstractFloat})
"""
function myldiv!(Y::AbstractVector, tmp::FDFactorTmp{V}, A::FDFactor, b::AbstractVector{<:AbstractFloat}) where {V}
    m = LinearAlgebra.checksquare(factor(A))
    m == length(b) || throw(DimensionMismatch())
    m == length(Y) || throw(DimensionMismatch())
    
    # type signatures
    N = npartials(A)
    T = tagtype(A)
    
    Af = factor(A)
    tmp_real = Ar⁻¹y(tmp)
    ldiv!(tmp_real, Af, b)  # real part

    partialmat = tmp.partials_out
    Apj = tmp.partials_in

    for j in 1:N
        fill_partials!(tmp, A, j)
        tmp_part = view(partialmat, :, j)
        mul!(tmp_part, Apj, tmp_real, -1, false)
        ldiv!(Af, tmp_part)
    end

    for i in eachindex(Y)
        val = tmp_real[i]
        tup = NTuple{N,V}(view(partialmat, i, :))
        part = Partials(tup)
        Y[i] = Dual{T,V,N}(val, part)
    end
    return Y
end

# import LinearAlgebra.ldiv!


"""
    \\(M::FDFactor, y::AbstractVecor{<:AbstractFloat})

Backsubstitution for `FDFactor`.
See `DualFactors` for details.

    M-1 = (I - ε A-1 B) A-1
"""
function \(A::FDFactor, b::AbstractVector{<:AbstractFloat})
    
    # type signatures
    T = tagtype(A)
    N = npartials(A)    
    V = valtype(eltype(partials(A)))
    
    m = length(b)
    
    Ar = factor(A)
    Ar⁻¹y = Ar \ b  # real part

    partialmat = Matrix{V}(undef, m, N)
    tmp        = Vector{V}(undef, m)
    outvec     = Vector{Dual{T,V,N}}(undef, m)


    for j in 1:N
        Apj = partials(A,j)
        mul!(tmp, Apj, Ar⁻¹y, -1, false)
        partialj = view(partialmat, :,j)
        ldiv!(partialj, Ar, tmp)
    end

    for j in eachindex(outvec)
        val = Ar⁻¹y[j]
        tup = NTuple{N,V}(view(partialmat, j,:))
        part = Partials(tup)
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
