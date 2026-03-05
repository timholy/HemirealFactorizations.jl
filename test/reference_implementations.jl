using HemirealNumbers
using HemirealFactorizations

struct HemiCholeskyRealRef{T} <: HemirealFactorizations.AbstractHemiCholesky{T}
    Lreal::Matrix{T}
    r::Vector{T}
end

Base.size(F::HemiCholeskyRealRef) = size(F.Lreal)

function cholesky_reference(A::AbstractMatrix{<:Real})
    Base.require_one_based_indexing(A)
    T = float(eltype(A))
    n = size(A, 1)
    L = zeros(T, n, n)
    r = zeros(T, n)
    A = copyto!(similar(A, T), A)
    for j in 1:n
        Ajj = A[j,j]
        Aijmax = zero(T)
        for i in j+1:n
            Aijmax = max(Aijmax, abs(A[i,j]))
        end
        r[j] = rj = sign(Ajj) * min(abs(Ajj) / Aijmax, 1)
        L[j,j] = sqrt(Ajj / (2 * rj))   # ν component
        for i in j+1:n
            L[i,j] = A[i,j] / (2 * rj * L[j,j])  # μ component
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
