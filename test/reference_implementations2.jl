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
        rj = oneunit(T)
        for i in j+1:n
            Aij = A[i,j]
            rij = ratio0(abs(Ajj * A[i, i]), Aij^2)
            rj = min(rj, rij)
        end
        rj *= sign(Ajj)
        @show rj
        r[j] = rj
        L[j,j] = Ljj = iszero(rj) ? oneunit(T) : sqrt(Ajj / (2 * rj))   # ν component
        den = (1 + rj^2) * Ljj
        for i in j+1:n
            L[i,j] = A[i,j] / den  # μ component
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

function HemirealFactorizations._getL(F::HemiCholeskyRealRef{T}, i::Integer, j::Integer) where T
    rj, Lij = F.r[j], F.Lreal[i, j]
    return i == j ? PureHemi{T}(rj * Lij, Lij) : (i < j ? zero(PureHemi{T}) : PureHemi{T}(Lij, rj * Lij))
end

function mydiv(F::HemiCholeskyRealRef{T}, b::AbstractVector{T}, s) where T
    y = similar(b, PureHemi{T})
    for i in eachindex(y)
        delta = zero(T)
        for j in 1:i-1
            rj = F.r[j]
            delta += F.Lreal[i,j] * ((1 + rj^2) * y[j].n + rj * (1 - rj^2) * s[j])
        end
        ri, Lii, si = F.r[i], F.Lreal[i,i], s[i]
        @show b[i] delta Lii ri si
        yiνri = (b[i] - delta - Lii * (1 - ri^2) * si) / (2 * Lii)
        y[i] = PureHemi{T}(ri * yiνri + (1 - ri^2) * si, yiνri / ri)
    end
    @show y
    x = similar(b, T)
    for i in reverse(eachindex(y))
        delta = zero(PureHemi{T})
        for j in i+1:length(y)
            delta += HemirealFactorizations._getL(F, j, i) * x[j]
        end
        num = y[i] - delta
        denom = HemirealFactorizations._getL(F, i, i)
        @show num denom
        x[i] = [denom.m, denom.n] \ [num.m, num.n]  # least-squares solve
    end
    return x
end

function calcs(F::HemiCholeskyRealRef{T}, x) where T
    s = fill!(similar(x, T), 0)
    for i in eachindex(s)
        stmp = s[i]
        for j in i+1:length(x)
            Ljiμ = F.Lreal[j,i]
            stmp += Ljiμ * x[j]
        end
        s[i] = stmp
    end
    return s
end

function mydiv(F::HemiCholeskyRealRef{T}, b::AbstractVector{T}, w) where T
    y = similar(b, PureHemi{T})
    # Forward substitution to solve L * y = b
    for i in eachindex(y)
        gi = b[i]
        for j in 1:i-1
            Lij = HemirealFactorizations._getL(F, i, j)
            gi -= Lij * y[j]
        end
        ri, Liiν, wi = F.r[i], F.Lreal[i,i], w[i]
        yiμ = -gi / (Liiν * (1 + ri * wi))
        y[i] = PureHemi{T}(yiμ, yiμ * wi)
    end
    x = similar(b, T)
    for i in reverse(eachindex(y))
        hi = y[i]
        for j in i+1:length(y)
            hi -= conj(HemirealFactorizations._getL(F, j, i)) * x[j]
        end
    end
    return x
end
