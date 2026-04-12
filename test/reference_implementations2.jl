using HemirealNumbers
using HemirealFactorizations

struct HemiCholeskyRealRef{T} <: HemirealFactorizations.AbstractHemiCholesky{T}
    Lreal::Matrix{T}
    r::Vector{T}
end

Base.size(F::HemiCholeskyRealRef) = size(F.Lreal)

ratio0(num, denom) = iszero(num) ? num / oneunit(denom) : num / denom

function cholesky_reference(A::AbstractMatrix{<:Real})
    Base.require_one_based_indexing(A)
    T = float(eltype(A))
    n = size(A, 1)
    L = zeros(T, n, n)
    r = zeros(T, n)
    A = copyto!(similar(A, T), A)
    for j in 1:n
        Ajj = A[j,j]
        rj = zero(T)
        for i in j+1:n
            Aij = A[i,j]
            rj += exp(-ratio0(Aij^2, abs(Ajj * A[i, i])))   # each contributes a value in [0, 1]; it's 0 when Aij is big relative to √(Ajj * A[i, i]) and 1 when Aij is small relative to √(Ajj * A[i, i])
        end
        if j < n
            rj /= n - j
        else
            rj = oneunit(T)
        end
        r[j] = rj
        L[j,j] = Ljj = iszero(rj) ? oneunit(T) : sqrt(Ajj / (2 * rj))   # ν component
        for i in j+1:n
            L[i,j] = A[i,j] / ((1 + rj^2) * Ljj)  # μ component
        end
        for k in j+1:n
            for i in k:n
                A[i,k] -= 2 * rj * L[i,j] * L[k,j]
            end
        end
    end
    return HemiCholeskyRealRef(L, r)
end

function Base.getproperty(F::HemiCholeskyRealRef, sym::Symbol)
    if sym === :L
        L = similar(F.Lreal, PureHemi{eltype(F.Lreal)})
        fill!(L, PureHemi{eltype(F.Lreal)}(0, 0))
        for j in 1:size(F.Lreal, 2)
            Ljjν = F.Lreal[j,j]
            rj = F.r[j]
            L[j,j] = PureHemi{eltype(F.Lreal)}(rj * Ljjν, Ljjν)
            for i in j+1:size(F.Lreal, 1)
                Lijμ = F.Lreal[i,j]
                L[i,j] = PureHemi{eltype(F.Lreal)}(Lijμ, rj * Lijμ)
            end
        end
        return L
    else
        return getfield(F, sym)
    end
end

HemirealFactorizations.hrmatrix(::Type{T}, F::HemiCholeskyRealRef{T}) where T = F.L
