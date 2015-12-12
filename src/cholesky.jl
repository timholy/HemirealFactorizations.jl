import Base: *, \, unsafe_getindex
using Base.BLAS: syr!, ger!, syrk!, syr2k!
using Base.LinAlg: BlasFloat

abstract AbstractHemiCholesky{T<:Real}

immutable HemiCholesky{T} <: AbstractHemiCholesky{T}
    L::Matrix{PureHemi{T}}
end
# Pure-hemi encoded as real (stores the nu-component in lower-triangle of L)
immutable HemiCholeskyReal{T} <: AbstractHemiCholesky{T}
    L::Matrix{T}
    d::Vector{Int8}
end

immutable HemiCholeskyXY{T<:Real,Ltype<:AbstractHemiCholesky,Htype} <: AbstractHemiCholesky{T}
    L::Ltype
    X::Matrix{T}
    Y::Matrix{PureHemi{T}}
    H::Htype
    Q::Matrix{T}
    nullindex::Vector{Int}
end
HemiCholeskyXY(L::AbstractHemiCholesky, X, Y, H, Q, nullindex) = HemiCholeskyXY{eltype(X), typeof(L), typeof(H)}(L, X, Y, H, Q, nullindex)

for FT in (HemiCholesky, HemiCholeskyReal, HemiCholeskyXY)
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
Base.convert{T}(::Type{HemiCholesky{T}}, F::HemiCholeskyXY) = convert(HemiCholesky{T}, F.L)

Base.convert{T}(::Type{Matrix}, F::AbstractHemiCholesky{T}) = convert(HemiCholesky{T}, F).L
Base.convert{T}(::Type{Matrix{T}}, F::AbstractHemiCholesky) = convert(HemiCholesky{T}, F).L

function Base.show(io::IO, F::AbstractHemiCholesky)
    println(io, Base.dims2string(size(F)), " ", typeof(F), ':')
    _show(io, convert(Matrix, F))
end

_show(io::IO, M::Matrix) = Base.with_output_limit(()->Base.print_matrix(io, M))


function Base.A_mul_Bt(F1::AbstractHemiCholesky, F2::AbstractHemiCholesky)
    L = convert(Matrix, F1)
    if F1 === F2
        return L*L'
    end
    L2 = convert(Matrix, F2)
    return L1*L2'
end
Base.A_mul_Bc(F1::AbstractHemiCholesky, F2::AbstractHemiCholesky) = A_mul_Bt(F1, F2)

Base.rank(F::HemiCholeskyXY) = size(F,1) - length(F.nullindex)

function Base.cholfact{T}(::Type{PureHemi{T}}, A::AbstractMatrix, δ=defaultδ(A); blocksize=default_blocksize(T))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    A0 = Array(floattype(T), size(A))
    copy!(A0, A)
    cholfact!(PureHemi{T}, A0, δ; blocksize=blocksize)
end
Base.cholfact(::Type{PureHemi}, A::AbstractMatrix, δ=defaultδ(A); blocksize=default_blocksize(floattype(eltype(A)))) = cholfact(PureHemi{floattype(eltype(A))}, A, δ; blocksize=blocksize)

# Blocked, cache-friendly algorithm
function Base.cholfact!{T<:AbstractFloat}(::Type{PureHemi{T}}, A::AbstractMatrix{T}, δ=defaultδ(A); blocksize=default_blocksize(T))
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
        solve_diagonal!(B11, d1, δ)
        if jend < K
            B21 = sub(A, jend+1:K, j:jend)
            solve_columns!(B21, d1, B11)
            B22 = sub(A, jend+1:K, jend+1:K)
            update_columns!(B22, d1, B21)
        end
    end
    L = HemiCholeskyReal(A, d)
    X, Y, H, Q, nullindex = solve_singularities(L, d)
    HemiCholeskyXY(L, X, Y, H, Q, nullindex)
end
Base.cholfact!{T<:AbstractFloat}(::Type{PureHemi}, A::AbstractMatrix{T}, δ=defaultδ(A); blocksize=default_blocksize(T)) = cholfact!(PureHemi{T}, A, δ; blocksize=blocksize)

function solve_diagonal!(A, d, δ)
    K = size(A, 1)
    for j = 1:K
        Ajj = A[j,j]
        if abs(Ajj) > δ
            # compute ℓ (as the jth column of A)
            d[j] = sign(Ajj)
            s = sqrt(2*abs(Ajj))
            A[j,j] = s/2
            f = d[j]/s
            for i = j+1:K
                A[i,j] *= f
            end
            update_columns!(sub(A, j+1:K, j+1:K), d[j], slice(A, j+1:K, j))
        else
            d[j] = 0
            A[j,j] = 0
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

function solve_singularities(L, d)
    ns = sum(d .== 0)
    K = size(L, 1)
    T = real(eltype(L))
    X = Array(T, K, ns)
    H = Array(T, ns, ns)
    Y = Array(PureHemi{T}, K, ns)
    ns == 0 && return X, Y, H, Array(T, K, 0), Int[]
    forwardsubst!(Y, L)
    backwardsubst!(X, H, L, Y)
    Hf = svdfact(H)
    # the dimensionality of the null space is the number of singular
    # values we discard
    s = Hf[:S]
    tol = defaultδ(X)
    nullrank = 0
    for i = 1:length(s)
        if s[i] < tol
            s[i] = 0
            nullrank += 1
        end
    end
    if nullrank == 0
        Q = similar(X, K, 0)
        nullindex = Int[]
    else
        # Identify the columns of X in the null space
        if nullrank == size(X, 2)
            nullindex = collect(1:size(X,2))
        else
            LtX = nucomponent(L, X)
            LtXmax = maxabs(LtX, 1)
            p = sortperm(squeeze(LtXmax, 1))
            nullindex = p[1:nullrank]
        end
        Q, _ = qr(X[:,nullindex])
    end
    X, Y, Hf, Q, nullindex
end

function (\){T}(F::HemiCholeskyXY{T}, b::AbstractVector)
    L = F.L
    K = length(b)
    size(L,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(L)) of the matrix"))
    ytilde = Array(PureHemi{T}, K)
    if isempty(F.nullindex)
        forwardsubst!(ytilde, L, b)
    else
        # project out the component of b perpendicular to the null space
        bproj = F.Q'*b
        forwardsubst!(ytilde, L, b - F.Q*bproj)
    end
    ns = size(F.X, 2)
    htilde = Array(T, ns)
    xtilde = Array(T, K)
    backwardsubst!(xtilde, htilde, L, ytilde)
    ns == 0 && return xtilde
    α = resolve(F.H, htilde)
    x = xtilde+F.X*α
    # Return the least-squares answer
    if !isempty(F.nullindex)
        xproj = F.Q'*x
        x = x - F.Q*xproj
    end
    x
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

function resolve(H, htildeα)
    resolve(H[:U], H[:S], H[:Vt], htildeα)
end

@noinline function resolve{T}(U::AbstractMatrix{T}, s::AbstractVector{T}, Vt::AbstractMatrix{T}, htildeα::AbstractVector{T})
    sinv = similar(s)
    for i = 1:length(s)
        si = s[i]
        sinv[i] = si > 0 ? -1/si : zero(si)   # solving with -htildeα
    end
    Vt'*(sinv .* (U'*htildeα))
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
singular_diagonals(L::HemiCholeskyReal) = find(L.d .== 0)

# Compute the nu-component of L'*X on the rows with singular diagonals
function nucomponent(L, X)
    indxsing = singular_diagonals(L)
    K, nc = size(X, 1), size(X, 2)
    LtX = zeros(eltype(X), nc, nc)
    for (ji,j) in enumerate(indxsing)
        for k = 1:nc
            tmp = LtX[ji,k]
            for i = j+1:K
                tmp += L.L[i,j]*X[i,k]
            end
            LtX[ji,k] = tmp
        end
    end
    LtX
end

floattype{T<:AbstractFloat}(::Type{T}) = T
floattype{T<:Integer}(::Type{T}) = Float64

const cachesize = 2^15

defaultδ(A) = 100 * size(A, 1) * eps(floattype(eltype(A))) * maxabs(A)
default_blocksize{T}(::Type{T}) = max(4, floor(Int, sqrt(cachesize/sizeof(T)/4)))
