using ForwardDiff: value, partials, Dual, Partials
import Base:\

# mabye also look at 
# https://github.com/briochemc/DualMatrixTools.jl/blob/master/src/DualMatrixTools.jl

function \(A::SparseMatrixCSC{Dual{T, V, N}, P}, b::AbstractVector{G}) where {T, V, N, P<:Integer, G}
    # T: tagtype(eltype(A))
    # V: valtype(eltype(A))
    # N: npartials(eltype(A))
    println("invoked sparse one")
    return __FDbackslash(A, b, T, V, N)
end

tmpobj

function __FDbackslash(A, b, T, V, N)
    Areal = value.(A)
    breal = value.(b)
    outreal = Areal\breal
    M = length(outreal)
    deriv = zeros(V, M, N)
    for i in 1:N
        pAi = partials.(A, i)
        pbi = partials.(b, i)
        deriv[:, i] = -Areal\(pAi * outreal - pbi)
    end
    out = Vector{eltype(A)}(undef, M)
    for j in eachindex(out)
        out[j] = ForwardDiff.Dual{T}(outreal[j], ForwardDiff.Partials(tuple(deriv[j,:]...)))
    end
    return out
end