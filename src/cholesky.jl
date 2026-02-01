import Base: *, \, unsafe_getindex
using LinearAlgebra.BLAS: syr!, ger!, syrk!, syr2k!
using LinearAlgebra: BlasFloat, AdjointFactorization, TransposeFactorization

export nullsolver

### Types, conversions, and basic utilities

abstract type AbstractHemiCholesky{T} <: Factorization{PureHemi{T}} end

struct HemiCholesky{T<:Real} <: AbstractHemiCholesky{T}
    L::Matrix{PureHemi{T}}
end

# Pure-hemi encoded as real (stores the nu-component in lower-triangle of L)
struct HemiCholeskyReal{T<:Real} <: AbstractHemiCholesky{T}
    L::Matrix{T}
    d::Vector{Int8}  # diagonal sign (-1, 0, or 1)
end

struct HemiCholeskyPivot{T<:Real} <: AbstractHemiCholesky{T}
    L::HemiCholeskyReal{T}
    piv::Vector{Int}
end

struct HemiCholeskyXY{T<:Real,Ltype<:AbstractHemiCholesky,Htype} <: AbstractHemiCholesky{T}
    L::Ltype
    X::Matrix{T}
    Y::Matrix{PureHemi{T}}
    HF::Htype
    Q::Matrix{T}
    nullflag::BitVector
end
HemiCholeskyXY(L::HemiCholeskyReal) = nullsolver(L)

const TransposedAbstractHemiCholesky{T} = Union{AdjointFactorization{PureHemi{T}, <:AbstractHemiCholesky{T}}, TransposeFactorization{PureHemi{T}, <:AbstractHemiCholesky{T}}}

function nullsolver(L::Union{HemiCholeskyReal,HemiCholeskyPivot}; tol=default_tol(L))
    X, Y, HF, Q, nullflag = solve_singularities(L; tol=tol)
    HemiCholeskyXY{eltype(X), typeof(L), typeof(HF)}(L, X, Y, HF, Q, nullflag)
end

for FT in (HemiCholesky, HemiCholeskyReal, HemiCholeskyPivot, HemiCholeskyXY)
    @eval begin
        Base.size(F::$FT) = size(F.L)
        Base.size(F::$FT, d::Integer) = size(F.L, d)
        Base.eltype(::Type{$FT{T}}) where T = PureHemi{T}
    end
end
# Base.eltype(::Type{HemiCholeskyXY{T,Ltype}}) where {T,Ltype} = PureHemi{T}

Base.getindex(F::HemiCholesky, i::Integer, j::Integer) = F.L[i,j]

function unsafe_getindex(F::HemiCholeskyReal{T}, i::Integer, j::Integer) where T
    d = F.d[j]
    nu = F.L[i,j]
    ifelse(d == 0 && i==j, PureHemi{T}(1,0), PureHemi{T}(d*nu, nu))
end
function Base.getindex(F::HemiCholeskyReal{T}, i::Integer, j::Integer) where T
    ifelse(i >= j, Base.unsafe_getindex(F, i, j), PureHemi{T}(0, 0))
end

hrmatrix(::Type{T}, F::HemiCholesky) where T = convert(Matrix{PureHemi{T}}, F.L)
function hrmatrix(::Type{T}, F::HemiCholeskyReal) where T
    L = Array{PureHemi{T}}(undef, size(F))
    K = size(F, 1)
    for j = 1:K
        for i = 1:j-1
            L[i,j] = zero(PureHemi{T})
        end
        for i = j:K
            L[i,j] = F[i,j]
        end
    end
    L
end
hrmatrix(::Type{T}, F::HemiCholeskyPivot) where T = hrmatrix(T, F.L)
hrmatrix(::Type{T}, F::HemiCholeskyXY) where T = hrmatrix(T, F.L)

hrmatrixpiv(::Type{T}, F::HemiCholesky) where T      = hrmatrix(T, F)
hrmatrixpiv(::Type{T}, F::HemiCholeskyReal) where T  = hrmatrix(T, F)
hrmatrixpiv(::Type{T}, F::HemiCholeskyPivot) where T = hrmatrix(T, F)[invperm(F.piv),:]
hrmatrixpiv(::Type{T}, F::HemiCholeskyXY) where T    = hrmatrixpiv(T, F.L)

Base.convert(::Type{HemiCholesky{T}}, F::HemiCholesky) where T = hrmatrix(T, F)
Base.convert(::Type{HemiCholesky{T}}, F::HemiCholeskyReal) where T = hrmatrix(T, F)
Base.convert(::Type{HemiCholesky{T}}, F::HemiCholeskyXY) where T = convert(HemiCholesky{T}, F.L)
Base.convert(::Type{HemiCholesky}, F::AbstractHemiCholesky{T}) where T = convert(HemiCholesky{T}, F)

Base.convert(::Type{Matrix}, F::AbstractHemiCholesky{T}) where T = hrmatrixpiv(T, F)
Base.convert(::Type{Matrix{T}}, F::AbstractHemiCholesky) where T = hrmatrixpiv(T, F)

function Base.show(io::IO, ::MIME"text/plain", F::AbstractHemiCholesky)
    println(io, Base.dims2string(size(F)), " ", typeof(F), ':')
    _show(io, F)
end
_show(io::IO, F::AbstractHemiCholesky{T}) where T = Base.print_matrix(IOContext(io, :limit => true), hrmatrix(T, F))
function _show(io::IO, F::HemiCholeskyPivot{T}) where T
    Base.print_matrix(IOContext(io, :limit => true), hrmatrix(T, F))
    println(io, "\n  pivot: ", F.piv)
end
_show(io::IO, F::HemiCholeskyXY{T}) where T = _show(io, F.L)

function (*)(F1::AbstractHemiCholesky, F2::TransposedAbstractHemiCholesky)
    L1 = convert(Matrix, F1)
    F2 = parent(F2)
    if F1 === F2
        return L1*L1'
    end
    L2 = convert(Matrix, F2)
    return L1*L2'
end

# Base.full(F::AbstractHemiCholesky) = F*F'

LinearAlgebra.rank(F::HemiCholeskyXY) = size(F,1) - size(F.Q,2)
function LinearAlgebra.rank(F::AbstractHemiCholesky)
    nzeros = nzerodiags(F)
    if nzeros != 0
        error("Cannot compute rank where there are zero diagonals;\n compute rank on the output of `nullsolver(F)`")
    end
    size(F,1)
end

### Computing the factorization of a matrix

function LinearAlgebra.cholesky(::Type{PureHemi{T}}, A::AbstractMatrix, pivot=Val{false}; tol=default_tol(A), blocksize=default_blocksize(T)) where T
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    A0 = Array{floattype(T)}(undef, size(A))
    copy!(A0, A)
    cholesky!(PureHemi{T}, A0, pivot; tol=tol, blocksize=blocksize)
end
LinearAlgebra.cholesky(::Type{PureHemi}, A::AbstractMatrix, pivot=Val{false}; tol=default_tol(A), blocksize=default_blocksize(floattype(eltype(A)))) = cholesky(PureHemi{floattype(eltype(A))}, A, pivot; tol=tol, blocksize=blocksize)

# Blocked, cache-friendly algorithm
function LinearAlgebra.cholesky!(::Type{PureHemi{T}}, A::AbstractMatrix{T}, pivot::Type{Val{false}}=Val{false}; tol=default_tol(A), blocksize=default_blocksize(T)) where T
    size(A,1) == size(A,2) || error("A must be square")
    eltype(A)<:Real || error("element type $(eltype(A)) not yet supported")
    K = size(A, 1)
    d = Vector{Int8}(undef, K)
    for j = 1:blocksize:K
        # Split A into
        #            |
        #       B11  |
        #            |
        # A = ----------------
        #            |
        #       B21  |   B22
        #            |
        jend = min(K, j+blocksize-1)
        B11 = view(A, j:jend, j:jend)
        d1 = view(d, j:jend)
        solve_diagonal!(B11, d1, tol)
        if jend < K
            B21 = view(A, jend+1:K, j:jend)
            solve_columns!(B21, d1, B11)
            B22 = view(A, jend+1:K, jend+1:K)
            update_columns!(B22, d1, B21)
        end
    end
    HemiCholeskyReal(A, d)
end

# Version with pivoting
function LinearAlgebra.cholesky!(::Type{PureHemi{T}}, A::AbstractMatrix{T}, pivot::Type{Val{true}}; tol=default_tol(A), blocksize=default_blocksize(T)) where T<:AbstractFloat
    size(A,1) == size(A,2) || error("A must be square")
    eltype(A)<:Real || error("element type $(eltype(A)) not yet supported")
    K = size(A, 1)
    d = Vector{Int8}(undef, K)
    piv = collect(1:K)
    Ad = diag(A)
    for j = 1:blocksize:K
        jend = min(K, j+blocksize-1)
        solve_columns_pivot!(A, d, piv, Ad, tol, j:jend)
        if jend < K
            d1 = view(d, j:jend)
            B21 = view(A, jend+1:K, j:jend)
            B22 = view(A, jend+1:K, jend+1:K)
            update_columns!(B22, d1, B21)
        end
    end
    HemiCholeskyPivot(HemiCholeskyReal(A, d), piv)
end


LinearAlgebra.cholesky!(::Type{PureHemi}, A::AbstractMatrix{T}, pivot=Val{false}; tol=default_tol(A), blocksize=default_blocksize(T)) where {T<:AbstractFloat} =
    cholesky!(PureHemi{T}, A; tol=tol, blocksize=blocksize)


function solve_diagonal!(B, d, tol)
    K = size(B, 1)
    for j = 1:K
        Bjj = B[j,j]
        if abs(Bjj) > tol
            # compute ℓ (as the jth column of B)
            d[j] = sign(Bjj)
            s = sqrt(2*abs(Bjj))
            B[j,j] = s/2
            f = d[j]/s
            for i = j+1:K
                B[i,j] *= f
            end
            # subtract ℓ[j+1:end]⊗ℓ[j+1:end] from the lower right quadrant
            update_columns!(view(B, j+1:K, j+1:K), d[j], view(B, j+1:K, j))
        else
            d[j] = 0
            B[j,j] = 0
            # ν^2 = 0, so this has no impact on the rest of the matrix
        end
    end
    B
end

# Here, pivoting applies to the whole matrix, so we don't pass in a view.
# The jrange input describes the columns we're supposed to handle now.
function solve_columns_pivot!(A, d, piv, Ad, tol, jrange)
    K, KA = last(jrange), size(A, 1)
    jmin = first(jrange)
    for j in jrange
        # Find the remaining diagonal with largest magnitude
        Amax = zero(eltype(A))
        jmax = j-1
        for jj = j:KA
            tmp = abs(Ad[jj])
            if tmp > Amax
                Amax = tmp
                jmax = jj
            end
        end
        if jmax > j
            pivot!(A, j, jmax)
            Ad[j], Ad[jmax] = Ad[jmax], Ad[j]
            piv[j], piv[jmax] = piv[jmax], piv[j]
        end
        Ajj = A[j,j]
        for k = jmin:j-1
            tmp = A[j,k]
            Ajj -= 2*d[k]*tmp*tmp
        end
        if abs(Ajj) > tol
            # compute ℓ (as the jth column of A)
            d[j] = sign(Ajj)
            s = sqrt(2*abs(Ajj))
            A[j,j] = s/2
            f = d[j]/s
        else
            d[j] = 0
            A[j,j] = 0
            f = one(eltype(A))
        end
        for k = jmin:j-1
            @inbounds ck = 2*d[k]*A[j,k]
            @simd for i = j+1:KA
                @inbounds A[i,j] -= ck*A[i,k]
            end
        end
        dj = d[j]
        @simd for i = j+1:KA
            @inbounds tmp = A[i,j]
            tmp *= f
            @inbounds A[i,j] = tmp
            @inbounds Ad[i] -= 2*dj*tmp*tmp
        end
    end
    A
end

function solve_columns!(B21, d, B11)
    I, J = size(B21)
    for j = 1:J
        dj = d[j]
        dj == 0 && continue
        s = 2*B11[j,j]
        f = dj/s
        for i = 1:I
            B21[i,j] *= f
        end
        update_columns!(view(B21, :, j+1:J), dj, view(B21, :, j), view(B11, j+1:J, j))
    end
    B21
end

# Computes dest -= d*c*c', in the lower diagonal
function update_columns!(dest::StridedMatrix{T}, d::Number, c::StridedVector{T}) where {T<:BlasFloat}
    syr!('L', convert(T, -2*d), c, dest)
end

# Computes dest -= d*x*y'
function update_columns!(dest::StridedMatrix{T}, d::Number, x::StridedVector{T}, y::StridedVector{T}) where {T<:BlasFloat}
    ger!(convert(T, -2*d), x, y, dest)
end

# Computes dest -= C*diagm(d)*C', in the lower diagonal
function update_columns!(dest::StridedMatrix{T}, d::AbstractVector, C::StridedMatrix{T}) where {T<:BlasFloat}
    isempty(d) && return dest
    # If d is homogeneous, we can use syr rather than syr2
    allsame = true
    d1 = d[1]
    for i = 2:length(d)
        allsame &= (d[i] == d1)
    end
    allsame && d1 == 0 && return dest
    if allsame
        syrk!('L', 'N', convert(T, -2*d1), C, one(T), dest)
    else
        Cd = C .* d'
        syr2k!('L', 'N', -one(T), C, Cd, one(T), dest)
    end
    dest
end

# Pure-julia fallbacks for the above routines
# Computes dest -= d*c*c', in the lower diagonal
function update_columns!(dest, d::Number, c::AbstractVector)
    K = length(c)
    for j = 1:K
        dcj = 2*d*c[j]
        @simd for i = j:K
            @inbounds dest[i,j] -= dcj*c[i]
        end
    end
    dest
end

# Computes dest -= d*x*y'
function update_columns!(dest, d::Number, x::AbstractVector, y::AbstractVector)
    I, J = size(dest)
    for j = 1:J
        dyj = 2*d*y[j]
        @simd for i = 1:I
            @inbounds dest[i,j] -= dyj*x[i]
        end
    end
    dest
end

# Computes dest -= C*diagm(d)*C', in the lower diagonal
function update_columns!(dest, d::AbstractVector, C::AbstractMatrix)
    Ct = C'
    Cdt = scale(2*d, Ct)
    K = size(dest, 1)
    nc = size(C, 2)
    for j = 1:K
        for i = j:K
            tmp = zero(eltype(dest))
            @simd for k = 1:nc
                @inbounds tmp += Ct[k,i]*Cdt[k,j]
            end
            @inbounds dest[i,j] -= tmp
        end
    end
    dest
end


### Solving linear systems

function solve_singularities(L; tol=default_tol(L))
    ns = nzerodiags(L)
    K = size(L, 1)
    T = real(eltype(L))
    X = Array{T}(undef, K, ns)
    H = Array{T}(undef, ns, ns)
    Y = Array{PureHemi{T}}(undef, K, ns)
    ns == 0 && return X, Y, lu!(H), Array{T}(undef, K, 0), falses(0)
    forwardsubst!(Y, L)
    backwardsubst!(X, H, L, Y)
    # Find the columns of X in the null space
    Hmax = maximum(abs, H; dims=2)
    nullflag = dropdims(Hmax; dims=2) .< tol
    # Find an orthonormal basis for the null space
    if sum(nullflag) == 0
        Q = Array{T}(undef, K, 0)
    else
        Q, _ = qr(X[:,nullflag])
    end
    # Prepare the solver for the non-null components of X
    HF = lu!(H[.!nullflag, .!nullflag])
    X, Y, HF, Q, nullflag
end
solve_singularities(L::HemiCholeskyPivot; tol=default_tol(L)) = solve_singularities(L.L; tol=tol)

function (\)(L::Union{HemiCholesky{T},HemiCholeskyReal{T},HemiCholeskyPivot{T}}, b::AbstractVector; forcenull::Bool=false) where T<:Real
    K = length(b)
    size(L,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(L)) of the matrix"))
    bp, Lp = pivot(L, b)
    nnull = nzerodiags(Lp)
    if nnull != 0 && !forcenull
        error("There were zero diagonals; use `nullsolver(L)\\b` or, if you're sure all zeros correspond to null directions, (\\)(L, b, forcenull=true)`.")
    end
    ytilde = Array{PureHemi{T}}(undef, K)
    forwardsubst!(ytilde, Lp, bp)
    xtilde = Array{T}(undef, K)
    htilde = Array{T}(undef, nnull)
    backwardsubst!(xtilde, htilde, Lp, ytilde)
    ipivot(L, xtilde)
end

function (\)(F::HemiCholeskyXY{T}, b::AbstractVector) where T
    L = F.L
    K = length(b)
    size(L,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(L)) of the matrix"))
    bp, Lp = pivot(L, b)
    ytilde = Array{PureHemi{T}}(undef, K)
    nnull = size(F.Q, 2)
    if nnull == 0
        forwardsubst!(ytilde, Lp, bp)
    else
        # project out the component of bp perpendicular to the null space
        bproj = F.Q'*bp
        forwardsubst!(ytilde, Lp, bp - F.Q*bproj)
    end
    ns = size(F.X, 2)
    htilde = Array{T}(undef, ns)
    xtilde = Array{T}(undef, K)
    backwardsubst!(xtilde, htilde, Lp, ytilde)
    ns == 0 && return ipivot(L, xtilde)
    keep = .!F.nullflag
    α = -(F.HF\htilde[keep])
    x = xtilde+F.X[:,keep]*α
    # Return the least-squares answer
    if nnull > 0
        xproj = F.Q'*x
        x = x - F.Q*xproj
    end
    ipivot(L, x)
end

# Forward-substitution with right hand side zero: find the
# symmetric-division nullspace of L
function forwardsubst!(Y, L::AbstractHemiCholesky)
    K, ns = size(Y, 1), size(Y, 2)
    T = real(eltype(Y))
    fill!(Y, zero(eltype(Y)))
    gα = Vector{T}(undef, ns)   # α-coefficient on current row
    si = 0              # number of singular columns processed so far
    for i = 1:K
        for jj = 1:si
            gα[jj] = 0
        end
        for j = 1:i-1
            Lij = unsafe_getindex(L, i, j)
            for jj = 1:si
                gα[jj] -= Lij*Y[j,jj]
            end
        end
        Lii = unsafe_getindex(L, i, i)
        if issingular(Lii)
            for jj = 1:si
                Y[i,jj] = PureHemi{T}(0, gα[jj]/Lii.m)
            end
            # Add a new singular column
            si += 1
            Y[i,si] = PureHemi{T}(1, 0)
        else
            for jj = 1:si
                Y[i,jj] = gα[jj]/Lii
            end
        end
    end
    Y
end

function forwardsubst!(ytilde::AbstractVector, L::AbstractHemiCholesky, b::AbstractVector)
    K = length(ytilde)
    length(b) == size(L, 1) == K || throw(DimensionMismatch("Sizes $(size(ytilde)), $(size(L)), and $(size(b)) do not match"))
    T = real(eltype(ytilde))
    for i = 1:K
        g = b[i]
        for j = 1:i-1
            Lij = unsafe_getindex(L, i, j)
            g -= Lij*ytilde[j]
        end
        Lii = unsafe_getindex(L, i, i)
        if issingular(Lii)
            ytilde[i] = PureHemi{T}(0, g/Lii.m)
        else
            ytilde[i] = g/Lii
        end
    end
    ytilde
end

function backwardsubst!(X, H, L::AbstractHemiCholesky, Y)
    K, nc = size(Y, 1), size(Y, 2)
    size(X, 1) == K && size(X, 2) == nc || throw(DimensionMismatch("Sizes $(size(X)) and $(size(Y)) of X and Y must match"))
    T = real(eltype(Y))
    h = Vector{PureHemi{T}}(undef, nc)  # the current row
    si = ns = size(H, 1)        # number of singular diagonals
    for i = K:-1:1
        for jj = 1:nc
            h[jj] = Y[i,jj]
        end
        for j = i+1:K
            Lji = unsafe_getindex(L, j, i)
            for jj = 1:nc
                h[jj] -= Lji*X[j,jj]
            end
        end
        Lii = unsafe_getindex(L, i, i)
        if issingular(Lii)
            for jj = 1:nc
                hjj = h[jj]
                X[i,jj] = hjj.m
                H[si,jj] = hjj.n
            end
            si -= 1
        else
            for jj = 1:nc
                X[i,jj] = h[jj].m/Lii.m
            end
        end
    end
    X
end

# Diagonal pivoting (row&column swap) for a lower triangular matrix
function pivot!(A, i::Integer, j::Integer)
    i, j = min(i,j), max(i,j)
    for k = 1:i-1
        A[i,k], A[j,k] = A[j,k], A[i,k]
    end
    A[i,i], A[j,j] = A[j,j], A[i,i]
    for k = i+1:j-1
        A[k,i], A[j,k] = A[j,k], A[k,i]
    end
    for k = j+1:size(A,1)
        A[k,i], A[k,j] = A[k,j], A[k,i]
    end
    A
end

pivot(L, b) = b, L
pivot(L::HemiCholeskyPivot, b) = b[L.piv], L.L

ipivot(L, b) = b
ipivot(L::HemiCholeskyPivot, b) = invpermute!(copy(b), L.piv)

issingular(x::PureHemi) = x.n == 0

function nzerodiags(L::HemiCholesky)
    ns = 0
    for i = 1:size(L,1)
        ns += issingular(L[i,i])
    end
    ns
end
function nzerodiags(L::HemiCholeskyReal)
    ns = 0
    for d in L.d
        ns += d == 0
    end
    ns
end
nzerodiags(L::HemiCholeskyPivot) = nzerodiags(L.L)
nzerodiags(L::HemiCholeskyXY) = size(L.X, 2)

function singular_diagonals(L)
    indxsing = Int[]
    for i = 1:size(L,1)
        if issingular(L[i,i])
            push!(indxsing, i)
        end
    end
    indxsing
end
singular_diagonals(L::HemiCholeskyReal) = find(L.d .== 0)

floattype(::Type{T}) where {T<:AbstractFloat} = T
floattype(::Type{T}) where {T<:Integer} = Float64

const cachesize = 2^15

default_δ(A) = 10 * size(A, 1) * eps(floattype(real(eltype(A))))
default_tol(A) = default_δ(A) * maximum(abs, A)
function default_tol(L::HemiCholeskyReal)
    K = size(L, 1)
    δ = default_δ(L.L)
    K == 0 && return δ
    ma = zero(eltype(L.L))
    for j = 1:K
        for i = j:K
            ma = max(ma, abs(L.L[i,j]))
        end
    end
    δ * ma
end
default_tol(L::HemiCholeskyPivot) = default_tol(L.L)
default_blocksize(::Type{T}) where {T} = max(4, floor(Int, sqrt(cachesize/sizeof(T)/4)))
