import Base: *, \, unsafe_getindex
using Base.BLAS: syr!, ger!, syrk!, syr2k!
using Base.LinAlg: BlasFloat

export nullsolver

### Types, conversions, and basic utilities

abstract AbstractHemiCholesky{T<:Real}

immutable HemiCholesky{T} <: AbstractHemiCholesky{T}
    L::Matrix{PureHemi{T}}
end

# Pure-hemi encoded as real (stores the nu-component in lower-triangle of L)
immutable HemiCholeskyReal{T} <: AbstractHemiCholesky{T}
    L::Matrix{T}
    d::Vector{Int8}  # diagonal sign (-1, 0, or 1)
end

immutable HemiCholeskyPivot{T} <: AbstractHemiCholesky{T}
    L::HemiCholeskyReal{T}
    piv::Vector{Int}
end

immutable HemiCholeskyXY{T<:Real,Ltype<:AbstractHemiCholesky,Htype} <: AbstractHemiCholesky{T}
    L::Ltype
    X::Matrix{T}
    Y::Matrix{PureHemi{T}}
    HF::Htype
    Q::Matrix{T}
    nullflag::BitVector
end
HemiCholeskyXY(L::HemiCholeskyReal) = nullsolver(L)

function nullsolver(L::Union{HemiCholeskyReal,HemiCholeskyPivot}; tol=default_tol(L))
    X, Y, HF, Q, nullflag = solve_singularities(L; tol=tol)
    HemiCholeskyXY{eltype(X), typeof(L), typeof(HF)}(L, X, Y, HF, Q, nullflag)
end

for FT in (HemiCholesky, HemiCholeskyReal, HemiCholeskyPivot, HemiCholeskyXY)
    @eval begin
        Base.size(F::$FT) = size(F.L)
        Base.size(F::$FT, d::Integer) = size(F.L, d)
        Base.eltype{T}(::Type{$FT{T}}) = PureHemi{T}
    end
end
Base.eltype{T,Ltype}(::Type{HemiCholeskyXY{T,Ltype}}) = PureHemi{T}

@inline Base.getindex(F::HemiCholesky, i::Integer, j::Integer) = F.L[i,j]

@inline function unsafe_getindex{T}(F::HemiCholeskyReal{T}, i::Integer, j::Integer)
    d = F.d[j]
    nu = F.L[i,j]
    ifelse(d == 0 && i==j, PureHemi{T}(1,0), PureHemi{T}(d*nu, nu))
end
@inline function Base.getindex{T}(F::HemiCholeskyReal{T}, i::Integer, j::Integer)
    ifelse(i >= j, Base.unsafe_getindex(F, i, j), PureHemi{T}(0, 0))
end

Base.convert{T}(::Type{HemiCholesky}, F::AbstractHemiCholesky{T}) = convert(HemiCholesky{T}, F)
function Base.convert{T}(::Type{HemiCholesky{T}}, F::HemiCholeskyReal)
    L = Array(PureHemi{T}, size(F))
    K = size(F, 1)
    for j = 1:K
        for i = 1:j-1
            L[i,j] = zero(PureHemi{T})
        end
        for i = j:K
            L[i,j] = F[i,j]
        end
    end
    HemiCholesky(L)
end
Base.convert{T}(::Type{HemiCholesky{T}}, F::HemiCholeskyPivot) = convert(HemiCholesky{T}, F.L)
Base.convert{T}(::Type{HemiCholesky{T}}, F::HemiCholeskyXY) = convert(HemiCholesky{T}, F.L)

Base.convert{T}(::Type{Matrix}, F::AbstractHemiCholesky{T}) = convert(HemiCholesky{T}, F).L
Base.convert{T}(::Type{Matrix{T}}, F::AbstractHemiCholesky) = convert(HemiCholesky{T}, F).L

function Base.show(io::IO, F::AbstractHemiCholesky)
    println(io, Base.dims2string(size(F)), " ", typeof(F), ':')
    Base.with_output_limit(()->Base.print_matrix(io, convert(Matrix, F)))
end
function Base.show(io::IO, F::HemiCholeskyPivot)
    println(io, Base.dims2string(size(F)), " ", typeof(F), ':')
    Base.with_output_limit(()->Base.print_matrix(io, convert(Matrix, F)))
    println(io, "\n  pivot: ", F.piv)
end

function Base.A_mul_Bt(F1::AbstractHemiCholesky, F2::AbstractHemiCholesky)
    L1 = convert(Matrix, F1)
    if F1 === F2
        return L1*L1'
    end
    L2 = convert(Matrix, F2)
    return L1*L2'
end
Base.A_mul_Bc(F1::AbstractHemiCholesky, F2::AbstractHemiCholesky) = A_mul_Bt(F1, F2)

Base.full(F::AbstractHemiCholesky) = F*F'

Base.rank(F::HemiCholeskyXY) = size(F,1) - size(F.Q,2)
function Base.rank(F::AbstractHemiCholesky)
    nzeros = nzerodiags(F)
    if nzeros != 0
        error("Cannot compute rank where there are zero diagonals;\n compute rank on the output of `nullsolver(F)`")
    end
    size(F,1)
end

### Computing the factorization of a matrix

function Base.cholfact{T}(::Type{PureHemi{T}}, A::AbstractMatrix, pivot=Val{false}; tol=default_tol(A), blocksize=default_blocksize(T))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    A0 = Array(floattype(T), size(A))
    copy!(A0, A)
    cholfact!(PureHemi{T}, A0, pivot; tol=tol, blocksize=blocksize)
end
Base.cholfact(::Type{PureHemi}, A::AbstractMatrix, pivot=Val{false}; tol=default_tol(A), blocksize=default_blocksize(floattype(eltype(A)))) = cholfact(PureHemi{floattype(eltype(A))}, A, pivot; tol=tol, blocksize=blocksize)

# Blocked, cache-friendly algorithm
function Base.cholfact!{T<:AbstractFloat}(::Type{PureHemi{T}}, A::AbstractMatrix{T}, pivot::Type{Val{false}}=Val{false}; tol=default_tol(A), blocksize=default_blocksize(T))
    size(A,1) == size(A,2) || error("A must be square")
    eltype(A)<:Real || error("element type $(eltype(A)) not yet supported")
    K = size(A, 1)
    d = Array(Int8, K)
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
        B11 = sub(A, j:jend, j:jend)
        d1 = sub(d, j:jend)
        solve_diagonal!(B11, d1, tol)
        if jend < K
            B21 = sub(A, jend+1:K, j:jend)
            solve_columns!(B21, d1, B11)
            B22 = sub(A, jend+1:K, jend+1:K)
            update_columns!(B22, d1, B21)
        end
    end
    HemiCholeskyReal(A, d)
end

# Version with pivoting
function Base.cholfact!{T<:AbstractFloat}(::Type{PureHemi{T}}, A::AbstractMatrix{T}, pivot::Type{Val{true}}; tol=default_tol(A), blocksize=default_blocksize(T))
    size(A,1) == size(A,2) || error("A must be square")
    eltype(A)<:Real || error("element type $(eltype(A)) not yet supported")
    K = size(A, 1)
    d = Array(Int8, K)
    piv = collect(1:K)
    blocksize = 1
    for j = 1:blocksize:K
        jend = min(K, j+blocksize-1)
        solve_diagonal_pivot!(A, d, piv, tol, j:jend)
        if jend < K
            B11 = sub(A, j:jend, j:jend)
            d1 = sub(d, j:jend)
            B21 = sub(A, jend+1:K, j:jend)
            solve_columns!(B21, d1, B11)
            B22 = sub(A, jend+1:K, jend+1:K)
            update_columns!(B22, d1, B21)
        end
    end
    HemiCholeskyPivot(HemiCholeskyReal(A, d), piv)
end


Base.cholfact!{T<:AbstractFloat}(::Type{PureHemi}, A::AbstractMatrix{T}, pivot=Val{false}; tol=default_tol(A), blocksize=default_blocksize(T)) = cholfact!(PureHemi{T}, A; tol=tol, blocksize=blocksize)



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
            update_columns!(sub(B, j+1:K, j+1:K), d[j], slice(B, j+1:K, j))
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
function solve_diagonal_pivot!(A, d, piv, tol, jrange)
    K, KA = last(jrange), size(A, 1)
    for j in jrange
        # Find the remaining diagonal with largest magnitude
        Amax = zero(eltype(A))
        jmax = j-1
        for jj = j:KA
            tmp = abs(A[jj,jj])
            if tmp > Amax
                Amax = tmp
                jmax = jj
            end
        end
        if jmax > j
            pivot!(A, j, jmax)
            piv[j], piv[jmax] = piv[jmax], piv[j]
        end
        Ajj = A[j,j]
        if abs(Ajj) > tol
            # compute ℓ (as the jth column of A)
            d[j] = sign(Ajj)
            s = sqrt(2*abs(Ajj))
            A[j,j] = s/2
            f = d[j]/s
            for i = j+1:K
                A[i,j] *= f
            end
            # subtract ℓ[j+1:end]⊗ℓ[j+1:end] from the lower right quadrant
            update_columns!(sub(A, j+1:K, j+1:K), d[j], slice(A, j+1:K, j))
        else
            d[j] = 0
            A[j,j] = 0
            # ν^2 = 0, so this has no impact on the rest of the matrix
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
        update_columns!(sub(B21, :, j+1:J), dj, slice(B21, :, j), slice(B11, j+1:J, j))
    end
    B21
end

# Computes dest -= d*c*c', in the lower diagonal
@inline function update_columns!{T<:BlasFloat}(dest::StridedMatrix{T}, d::Number, c::StridedVector{T})
    syr!('L', convert(T, -2*d), c, dest)
end

# Computes dest -= d*x*y'
@inline function update_columns!{T<:BlasFloat}(dest::StridedMatrix{T}, d::Number, x::StridedVector{T}, y::StridedVector{T})
    ger!(convert(T, -2*d), x, y, dest)
end

# Computes dest -= C*diagm(d)*C', in the lower diagonal
function update_columns!{T<:BlasFloat}(dest::StridedMatrix{T}, d::AbstractVector, C::StridedMatrix{T})
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
        if VERSION < v"0.5"
            Cd = scale(copy(C), copy(d))
        else
            Cd = scale(C, d)
        end
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
    X = Array(T, K, ns)
    H = Array(T, ns, ns)
    Y = Array(PureHemi{T}, K, ns)
    ns == 0 && return X, Y, lufact!(H), Array(T, K, 0), falses(0)
    forwardsubst!(Y, L)
    backwardsubst!(X, H, L, Y)
    # Find the columns of X in the null space
    Hmax = maxabs(H, 2)
    nullflag = squeeze(Hmax, 2) .< tol
    # Find an orthonormal basis for the null space
    if sum(nullflag) == 0
        Q = Array(T, K, 0)
    else
        Q, _ = qr(X[:,nullflag])
    end
    # Prepare the solver for the non-null components of X
    HF = lufact!(H[!nullflag, !nullflag])
    X, Y, HF, Q, nullflag
end
solve_singularities(L::HemiCholeskyPivot; tol=default_tol(L)) = solve_singularities(L.L; tol=tol)

function (\){T}(L::Union{HemiCholesky{T},HemiCholeskyReal{T},HemiCholeskyPivot{T}}, b::AbstractVector; forcenull::Bool=false)
    K = length(b)
    size(L,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(L)) of the matrix"))
    bp, Lp = pivot(L, b)
    nnull = nzerodiags(Lp)
    if nnull != 0 && !forcenull
        error("There were zero diagonals; use `nullsolver(L)\\b` or, if you're sure all zeros correspond to null directions, (\\)(L, b, forcenull=true)`.")
    end
    ytilde = Array(PureHemi{T}, K)
    forwardsubst!(ytilde, Lp, bp)
    xtilde = Array(T, K)
    htilde = Array(T, nnull)
    backwardsubst!(xtilde, htilde, Lp, ytilde)
    ipivot(L, xtilde)
end

function (\){T}(F::HemiCholeskyXY{T}, b::AbstractVector)
    L = F.L
    K = length(b)
    size(L,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(L)) of the matrix"))
    bp, Lp = pivot(L, b)
    ytilde = Array(PureHemi{T}, K)
    nnull = size(F.Q, 2)
    if nnull == 0
        forwardsubst!(ytilde, Lp, bp)
    else
        # project out the component of bp perpendicular to the null space
        bproj = F.Q'*bp
        forwardsubst!(ytilde, Lp, bp - F.Q*bproj)
    end
    ns = size(F.X, 2)
    htilde = Array(T, ns)
    xtilde = Array(T, K)
    backwardsubst!(xtilde, htilde, Lp, ytilde)
    ns == 0 && return ipivot(L, xtilde)
    keep = !F.nullflag
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
    gα = Array(T, ns)   # α-coefficient on current row
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
    length(b) == size(L, 1) == K || throw(DimensionMismatch("Sizes $(size(y)), $(size(L)), and $(size(b)) do not match"))
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
    h = Array(PureHemi{T}, nc)  # the current row
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
        A[i,k], A[j,k] = A[j,k], A[i,k]  # don't need this?
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
ipivot(L::HemiCholeskyPivot, b) = ipermute!(copy(b), L.piv)

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

floattype{T<:AbstractFloat}(::Type{T}) = T
floattype{T<:Integer}(::Type{T}) = Float64

const cachesize = 2^15

default_δ(A) = 10 * size(A, 1) * eps(floattype(real(eltype(A))))
default_tol(A) = default_δ(A) * maxabs(A)
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
default_blocksize{T}(::Type{T}) = max(4, floor(Int, sqrt(cachesize/sizeof(T)/4)))
