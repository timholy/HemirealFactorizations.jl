import Base: \

immutable PureHemiCholesky{T}
    L::Matrix{PureHemi{T}}
end
# Pure-hemi encoded as a real (stores the nu-component in lower-triangle of L)
immutable HemiCholeskyReal{T<:AbstractFloat}
    L::Matrix{T}
    d::Vector{Int8}
end

Base.size(F::PureHemiCholesky) = size(F.L)
Base.size(F::PureHemiCholesky, d::Integer) = size(F.L, d)
Base.size(F::HemiCholeskyReal) = size(F.L)
Base.size(F::HemiCholeskyReal, d::Integer) = size(F.L, d)

function Base.convert{T}(::Type{PureHemiCholesky{T}}, F::HemiCholeskyReal)
    L = Array(PureHemi{T}, size(F))
    K = size(F, 1)
    for j = 1:K
        Ljj, d = F.L[j,j], F.d[j]
        for i = 1:j-1
            L[i,j] = zero(PureHemi{T})
        end
        L[j,j] = d == 0 ? PureHemi{T}(1,0) : PureHemi{T}(Ljj*d, Ljj)
        for i = j+1:K
            Lij = F.L[i,j]
            L[i,j] = PureHemi{T}(d*Lij, Lij)
        end
    end
    PureHemiCholesky(L)
end
Base.convert{T}(::Type{PureHemiCholesky}, F::HemiCholeskyReal{T}) = convert(PureHemiCholesky{T}, F)

# Base.show(io::IO, F::HemiCholeskyReal) = show(io, convert(PureHemiCholesky, F))

# This could be done in-place with just real representations, but this
# simple implementation involves less bookkeeping.
function Base.cholfact{T}(::Type{PureHemi{T}}, A::AbstractMatrix, δ=defaultδ(A))
    issym(A) || error("A must be symmetric")
    eltype(A)<:Real || error("element type $(eltype(A)) not yet supported")
    L = fill(zero(PureHemi{T}), size(A))
    K = size(A, 1)
    for j = 1:K
        s = A[j,j]
        for k = 1:j-1
            l = L[j,k]
            s -= l*l
        end
        if abs(s) > δ
            f = sqrt(abs(s)/2)
            Ljj = PureHemi{T}(sign(s)*f, f)
            L[j,j] = Ljj
            for i = j+1:K
                s = A[i,j]
                for k = 1:j-1
                    s -= L[i,k]*L[j,k]
                end
                L[i,j] = s/Ljj
            end
        else
            L[j,j] = PureHemi{T}(1, 0)
            for i = j+1:K
                s = A[i,j]
                for k = 1:j-1
                    s -= L[i,k]*L[j,k]
                end
                L[i,j] = PureHemi{T}(0, s)
            end
        end
    end
    PureHemiCholesky(L)
end
Base.cholfact(::Type{PureHemi}, A::AbstractMatrix, δ=defaultδ(A)) = cholfact(PureHemi{floattype(eltype(A))}, A, δ)

# In-place and much higher performance (fairly cache-friendly)
# Could do even better by computing this in blocks
function Base.cholfact!{T<:AbstractFloat}(::Type{PureHemi{T}}, A::AbstractMatrix{T}, δ=defaultδ(A))
    size(A,1) == size(A,2) || error("A must be square")
    eltype(A)<:Real || error("element type $(eltype(A)) not yet supported")
    K = size(A, 1)
    d = Array(Int8, K)
    for j = 1:K
        Ajj = A[j,j]
        if abs(Ajj) > δ
            # compute ℓ (as A[j:k,j])
            d[j] = sign(Ajj)
            s = sqrt(2*abs(Ajj))
            A[j,j] = s/2
            f = d[j]/s
            for i = j+1:K
                A[i,j] *= f
            end
            # Subtract ℓ⊗ℓ from the rest of the matrix
            for k = j+1:K
                f = 2*d[j]*A[k,j]
                @simd for i = k:K
                    @inbounds A[i,k] -= A[i,j]*f
                end
            end
        else
            d[j] = 0
            A[j,j] = 0
        end
    end
    HemiCholeskyReal(A, d)
end
Base.cholfact!{T<:AbstractFloat}(::Type{PureHemi}, A::AbstractMatrix{T}, δ=defaultδ(A)) = cholfact!(PureHemi{T}, A, δ)

function (\){T}(F::PureHemiCholesky{T}, b::AbstractVector)
    L = F.L
    K = length(b)
    size(L,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(L)) of the matrix"))
    # Determine the singular diagonals
    indxsing = singular_diagonals(L)
    if isempty(indxsing)
        y = forwardsubst(L, b, indxsing, eltype(b)[])
    else
        # There were singular columns. We first have to determine α, the
        # μ-components of the forward-substitution solution for y in the
        # singular columns.
        ytilde, Y = forwardsubst(L, b, indxsing)
        xtilde, X, H, htildeα = backwardsubst(L, ytilde, Y, indxsing)
        α, Δb = resolve(H, htildeα, indxsing, b)
        y = forwardsubst(L, b+Δb, indxsing, α)
    end
    backwardsubst(L, y)
end

# Forward-substitution when there are no singular columns, or when you
# know the value α of the μ-component in the singular columns.
function forwardsubst{T}(L::Matrix{PureHemi{T}}, b, indxsing, α)
    K = length(b)
    y = fill(zero(PureHemi{T}), K)
    si = 0
    for i = 1:K
        r = b[i]
        for j = 1:i-1
            r -= L[i,j]*y[j]
        end
        Lii = L[i,i]
        if issingular(Lii)
            y[i] = PureHemi{T}(α[si+=1], r/Lii.m)
        else
            y[i] = r/Lii
        end
    end
    y
end

# Forward-substitution with placeholders for the undetermined
# μ-components α of the singular columns.
function forwardsubst{T}(L::Matrix{PureHemi{T}}, b, indxsing=singular_diagonals(L))
    K = length(b)
    ns = length(indxsing)
    ytilde = fill(zero(PureHemi{T}), K) # solution with α=0, y = ytilde + Y*α
    Y = fill(zero(PureHemi{T}), K, ns)  # coefficients of α on each row
    gα = Array(T, ns)                   # α-coefficient on current row
    si = 0 # number of singular columns processed so far
    for i = 1:K
        g = b[i]
        for jj = 1:si
            gα[jj] = 0
        end
        for j = 1:i-1
            Lij = L[i,j]
            g -= Lij*ytilde[j]
            for jj = 1:si
                gα[jj] -= Lij*Y[j,jj]
            end
        end
        Lii = L[i,i]
        if issingular(Lii)
            ytilde[i] = PureHemi{T}(0, g/Lii.m)
            for jj = 1:si
                Y[i,jj] = PureHemi{T}(0, gα[jj]/Lii.m)
            end
            # Add a new singular column
            si += 1
            Y[i,si] = PureHemi{T}(1, 0)  # add α[si]*μ to y[i]
        else
            ytilde[i] = g/Lii
            for jj = 1:si
                Y[i,jj] = gα[jj]/Lii
            end
        end
    end
    ytilde, Y
end

# Backward-substitution
function backwardsubst{T}(L::Matrix{PureHemi{T}}, y)
    K = length(y)
    x = Array(T, K)
    for i = K:-1:1
        h = y[i]
        for j = i+1:K
            Lji = L[j,i]
            h -= Lji*x[j]
        end
        Lii = L[i,i]
        if issingular(Lii)
            x[i] = h.m
        else
            x[i] = h/Lii
        end
    end
    x
end

# Backward-substitution with placeholders for the undetermined
# μ-components α of singular columns.
function backwardsubst{T}(L::Matrix{PureHemi{T}}, y, Y, indxsing=singular_diagonals(L))
    K = length(y)
    si = length(indxsing)
    ns = si
    xtilde = Array(T, K)
    X = zeros(T, K, si)
    H = Array(T, si, si)
    htildeα = Array(T, si)
    hα = Array(PureHemi{T}, si)
    ni = 0
    for i = K:-1:1
        htilde = y[i]
        for jj = 1:ns
            hα[jj] = Y[i,jj]
        end
        for j = i+1:K
            Lji = L[j,i]
            htilde -= Lji*xtilde[j]
            for jj = 1:ns
                hα[jj] -= Lji*X[j,jj]
            end
        end
        Lii = L[i,i]
        if issingular(Lii)
            xtilde[i] = htilde.m
            htildeα[si] = -htilde.n
            for jj = 1:ns
                h = hα[jj]
                H[si,jj] = h.n
                X[i,jj] = h.m
            end
            si -= 1
        else
            xtilde[i] = htilde/Lii
            for jj = 1:ns
                X[i,jj] = hα[jj]/Lii
            end
        end
    end
    xtilde, X, H, htildeα
end

function resolve(H, htildeα, indxsing, b)
    α = svdfact(H)\htildeα
    Δb = zeros(b)  # correction to b to put it in the range of A
    Δb[indxsing] = H*α - htildeα
    α, Δb
end

issingular(x::PureHemi) = x.n == 0

function singular_diagonals(L)
    indxsing = Int[]
    for i = 1:size(L,1)
        if issingular(L[i,i])
            push!(indxsing, i)
        end
    end
    indxsing
end

floattype{T<:AbstractFloat}(::Type{T}) = T
floattype{T<:Integer}(::Type{T}) = Float64

defaultδ(A) = sqrt(eps(floattype(eltype(A)))) * maxabs(A)
