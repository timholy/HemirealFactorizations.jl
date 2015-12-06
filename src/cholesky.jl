import Base: \

immutable HemirealCholesky{T}
    L::Matrix{PureHemi{T}}
end

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
            Ljj = PureHemi{T}(f, sign(s)*f)
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
                L[i,j] = PureHemi{T}(0, A[i,j])
            end
        end
    end
    HemirealCholesky(L)
end
Base.cholfact(::Type{PureHemi}, A::AbstractMatrix, δ=defaultδ(A)) = cholfact(PureHemi{floattype(eltype(A))}, A, δ)

function (\){T}(F::HemirealCholesky{T}, b::AbstractVector)
    L = F.L
    K = length(b)
    size(L,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(L)) of the matrix"))
    # Determine the singular diagonals
    indxsing = Int[]
    for i = 1:K
        if issingular(L[i,i])
            push!(indxsing, i)
        end
    end
    if isempty(indxsing)
        y = forwardsubst(L, b, indxsing, eltype(b)[])
    else
        # There were singular columns, we first have to determine α, the
        # μ-components of the forward-substitution solution for y in the
        # singular columns.
        y0, Yα = forwardsubst(L, b, indxsing)
        x, Xα, Cα, bα = backwardsubst(L, y0, Yα, indxsing)
        α, Δb = resolve(Cα, bα, indxsing, b)
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
function forwardsubst{T}(L::Matrix{PureHemi{T}}, b, indxsing)
    K = length(b)
    ns = length(indxsing)
    y0 = fill(zero(PureHemi{T}), K)      # solution with α=0, y = y0 + Yα*α
    Yα = fill(zero(PureHemi{T}), K, ns)  # coefficients of α on each row
    cα = Array(T, ns)                    # α-coefficient on current row
    si = 0 # number of singular columns processed so far
    for i = 1:K
        r = b[i]   # r = residual
        for jj = 1:si
            cα[jj] = 0
        end
        for j = 1:i-1
            Lij = L[i,j]
            r -= Lij*y0[j]
            for jj = 1:si
                cα[jj] -= Lij*Yα[j,jj]
            end
        end
        Lii = L[i,i]
        if issingular(Lii)
            y0[i] = PureHemi{T}(0, r/Lii.m)
            for jj = 1:si
                Yα[i,jj] = PureHemi{T}(0, cα[jj]/Lii.m)
            end
            # Add a new singular column
            si += 1
            Yα[i,si] = PureHemi{T}(1, 0)  # add α[si]*μ to y[i]
        else
            y0[i] = r/Lii
            for jj = 1:si
                Yα[i,jj] = cα[jj]/Lii
            end
        end
    end
    y0, Yα
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
            @assert h.n == h.m
            x[i] = h.m/Lii.m
        end
    end
    x
end

# Backward-substitution with placeholders for the undetermined
# μ-components α of singular columns.
function backwardsubst{T}(L::Matrix{PureHemi{T}}, y, Yα, indxsing)
    K = length(y)
    si = length(indxsing)
    ns = si
    x = Array(T, K)
    Xα = zeros(T, K, si)
    Cα = Array(T, si, si)
    bα = Array(T, si)
    hα = Array(PureHemi{T}, si)
    ni = 0
    for i = K:-1:1
        h = y[i]
        for jj = 1:ns
            hα[jj] = Yα[i,jj]
        end
        for j = i+1:K
            Lji = L[j,i]
            h -= Lji*x[j]
            for jj = 1:ns
                hα[jj] -= Lji*Xα[j,jj]
            end
        end
        Lii = L[i,i]
        if issingular(Lii)
            x[i] = h.m
            bα[si] = -h.n
            for jj = 1:ns
                h = hα[jj]
                Cα[si,jj] = h.n
                Xα[i,jj] = h.m
            end
            si -= 1
        else
            @assert h.n == h.m
            x[i] = h.m/Lii.m
            for jj = 1:ns
                Xα[i,jj] = hα[jj].m/Lii.m
            end
        end
    end
    x, Xα, Cα, bα
end

function resolve(Cα, bα, indxsing, b)
    α = svdfact(Cα)\bα
    Δb = zeros(b)  # correction to b to put it in the range of A
    Δb[indxsing] = Cα*α - bα
    α, Δb
end

issingular(x::PureHemi) = x.n == 0

floattype{T<:AbstractFloat}(::Type{T}) = T
floattype{T<:Integer}(::Type{T}) = Float64

defaultδ(A) = sqrt(eps(floattype(eltype(A)))) * maxabs(A)
