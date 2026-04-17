import Base: *, \
using LinearAlgebra.BLAS: syr!, ger!, syrk!, syr2k!
using LinearAlgebra: BlasFloat

### Types, conversions, and basic utilities

"""
    AbstractHemiCholesky{T<:Real} <: Factorization{PureHemi{T}}

Abstract supertype for Cholesky factorizations of real symmetric matrices over the hemireal
numbers. A hemireal Cholesky factorization of a real symmetric matrix `A` produces a
lower-triangular factor `L` with entries in [`PureHemi`](@ref) numbers such that `A = L * L'`.

Unlike the standard Cholesky factorization, hemireal Cholesky factorizations exist for *all*
real symmetric matrices: positive-definite, positive-semidefinite, indefinite, and singular.

Subtypes: [`HemiCholesky`](@ref), [`HemiCholeskyReal`](@ref), [`HemiCholeskyPivot`](@ref),
[`HemiCholeskyXY`](@ref).

See also [`cholesky`](@ref), [`cholesky!`](@ref), [`nullsolver`](@ref).
"""
abstract type AbstractHemiCholesky{T} <: Factorization{PureHemi{T}} end

"""
    HemiCholesky{T<:Real, S<:AbstractMatrix{PureHemi{T}}} <: AbstractHemiCholesky{T}

Matrix factorization type for the hemireal Cholesky factorization of a real symmetric matrix,
with the lower-triangular [`PureHemi`](@ref) factor stored directly as a matrix. This type is
rarely constructed directly; [`HemiCholeskyReal`](@ref) is the standard result type for real
inputs.

The lower-triangular factor is accessible as `F.L`. The original matrix is recovered by
`Matrix(F)`, satisfying `Matrix(F) ≈ A`.

Iterating the decomposition produces the components `L` and `U = L'`.

The following functions are available for `HemiCholesky` objects:
[`size`](@ref), [`\\`](@ref), [`Matrix`](@ref), [`det`](@ref), [`logdet`](@ref),
[`logabsdet`](@ref), [`isposdef`](@ref), [`issuccess`](@ref).
"""
struct HemiCholesky{T<:Real, S<:AbstractMatrix{PureHemi{T}}} <: AbstractHemiCholesky{T}
    L::S
end

# Pure-hemi encoded as real (stores the nu-component in lower-triangle of L)
"""
    HemiCholeskyReal{T<:Real, S<:AbstractMatrix{T}} <: AbstractHemiCholesky{T}

Matrix factorization type for the hemireal Cholesky factorization of a real
symmetric matrix `A`, using a compact real-number encoding of the
lower-triangular [`PureHemi`](@ref) factor. This is the return type of
[`cholesky`](@ref) and [`cholesky!`](@ref) for real input matrices.

The factorization computes a lower-triangular hemireal matrix `L_h` satisfying
`A = L_h * L_h'`. The ν-components of `L_h` are stored compactly in the internal
real matrix `F.Lreal`, and the sign of each diagonal entry is stored as `Int8`
in `F.d` (values in `{-1, 0, 1}`). The `(i,j)` entry of `L_h` for `i ≥ j` is
`PureHemi(F.d[j]*F.Lreal[i,j], F.Lreal[i,j])`.

The full lower-triangular [`PureHemi`](@ref) factor is accessible as `F.L`, and
its transpose as `F.U = F.L'` (the upper-triangular factor). A zero in `F.d`
indicates a zero pivot, which may or may not correspond to a singular direction.
The original matrix is recovered by `Matrix(F)`, satisfying `Matrix(F) ≈ A`.

Iterating the decomposition produces the components `L` and `U` in order.

The following functions are available for `HemiCholeskyReal` objects:
[`size`](@ref), [`\\`](@ref), [`ldiv!`](@ref), [`rdiv!`](@ref),
[`Matrix`](@ref), [`det`](@ref), [`logdet`](@ref), [`logabsdet`](@ref),
[`isposdef`](@ref), [`issuccess`](@ref), [`rank`](@ref), [`nullsolver`](@ref).
"""
struct HemiCholeskyReal{T<:Real, S<:AbstractMatrix{T}} <: AbstractHemiCholesky{T}
    Lreal::S          # compact real encoding: ν-components of the PureHemi factor
    d::Vector{Int8}   # diagonal sign (-1, 0, or 1)
end

"""
    HemiCholeskyPivot{T<:Real, S<:AbstractMatrix{T}} <: AbstractHemiCholesky{T}

Matrix factorization type for the pivoted hemireal Cholesky factorization of a real symmetric
matrix `A`. This is the return type of `cholesky(PureHemi, A, RowMaximum())`.

Diagonal pivoting reorders rows and columns to improve numerical stability. The inner
(unpivoted) factorization is stored in `F.L` as a [`HemiCholeskyReal`](@ref). The permutation
is accessible as `F.p` (vector) or `F.P` (matrix), with `F.piv` available for internal use.

The relationships satisfied are:
- `Matrix(F.L) ≈ A[F.p, F.p]` (unpivoted factor reconstructs the permuted matrix)
- `Matrix(F) ≈ A` (full reconstruction accounts for the permutation)

The upper-triangular factor is accessible as `F.U = F.L'`. Iteration is not supported
(consistent with `CholeskyPivoted` in `LinearAlgebra`).

The following functions are available for `HemiCholeskyPivot` objects:
[`size`](@ref), [`\\`](@ref), [`ldiv!`](@ref), [`rdiv!`](@ref), [`Matrix`](@ref),
[`det`](@ref), [`logdet`](@ref), [`logabsdet`](@ref), [`isposdef`](@ref),
[`issuccess`](@ref), [`rank`](@ref), [`nullsolver`](@ref).
"""
struct HemiCholeskyPivot{T<:Real, S<:AbstractMatrix{T}} <: AbstractHemiCholesky{T}
    F::HemiCholeskyReal{T, S}
    piv::Vector{Int}
end

"""
    HemiCholeskyXY{T<:Real, Ftype<:AbstractHemiCholesky, Htype} <: AbstractHemiCholesky{T}

Extended hemireal Cholesky factorization that augments a
[`HemiCholeskyReal`](@ref) or [`HemiCholeskyPivot`](@ref) with zero-pivot
handling, enabling stable least-squares solutions. Obtained from
[`nullsolver`](@ref).

When `F::HemiCholeskyXY`, `F \\ b` returns the minimum-norm least-squares
solution. The numerical rank is `rank(F)`.
"""
struct HemiCholeskyXY{T<:Real,Ftype<:AbstractHemiCholesky,Htype} <: AbstractHemiCholesky{T}
    F::Ftype
    X::Matrix{T}
    Y::Matrix{PureHemi{T}}
    HF::Htype
    Q::Matrix{T}
    nullflag::BitVector
end
HemiCholeskyXY(F::HemiCholeskyReal; tol=default_tol(F)) = nullsolver(F; tol)

"""
    nullsolver(F::Union{HemiCholeskyReal, HemiCholeskyPivot, SparseHemiCholeskyReal}; tol) -> HemiCholeskyXY

Augment a hemireal Cholesky factorization `F` with zero-pivot handling,
returning a [`HemiCholeskyXY`](@ref). This is needed for non-singular systems
with zero pivots, but it also handles singular systems in which case the result `Fs`
satisfies `Fs \\ b ≈ svd(A) \\ b` (minimum-norm least-squares solution).

The `tol` keyword controls the threshold for identifying null directions
(default: a multiple of machine epsilon scaled by the factor norm).

# Examples

```jldoctest
julia> A = [0 1; 1 0];   # nonsingular but with zero pivots

julia> F = cholesky(PureHemi, A)
2×2 HemiCholeskyReal{Float64, Matrix{Float64}}:
 1.0μ + 0.0ν  0.0μ + 0.0ν
 0.0μ + 1.0ν  1.0μ + 0.0ν

julia> F \\ [0.2, 0.3]
ERROR: There were zero diagonals; use `nullsolver(F)\\b` or, if you're sure all zeros correspond to null directions, `(\\)(F, b, forcenull=true)`.
Stacktrace:
[...]

julia> nullsolver(F) \\ [0.2, 0.3]
2-element Vector{Float64}:
 0.3
 0.2
```
"""
function nullsolver(F::Union{HemiCholeskyReal,HemiCholeskyPivot}; tol=default_tol(F))
    X, Y, HF, Q, nullflag = solve_zeropivots(F; tol)
    HemiCholeskyXY{eltype(X), typeof(F), typeof(HF)}(F, X, Y, HF, Q, nullflag)
end

Base.size(F::AbstractHemiCholesky) = size(F.L)
Base.size(F::AbstractHemiCholesky, d::Integer) = size(F.L, d)
Base.size(F::HemiCholeskyXY) = size(F.F)
Base.size(F::HemiCholeskyXY, d::Integer) = size(F.F, d)
Base.eltype(::Type{<:AbstractHemiCholesky{T}}) where T = PureHemi{T}

LinearAlgebra.issuccess(::AbstractHemiCholesky) = true
LinearAlgebra.isposdef(F::HemiCholeskyReal) = all(==(Int8(1)), F.d)
LinearAlgebra.isposdef(F::HemiCholeskyPivot) = isposdef(F.F)
LinearAlgebra.isposdef(F::HemiCholeskyXY) = isposdef(F.F)
LinearAlgebra.isposdef(F::HemiCholesky) = all(x -> x.m > 0 && x.n > 0, diag(F.L))

Base.copy(F::HemiCholesky) = HemiCholesky(copy(F.L))
Base.copy(F::HemiCholeskyReal) = HemiCholeskyReal(copy(F.Lreal), copy(F.d))
Base.copy(F::HemiCholeskyPivot) = HemiCholeskyPivot(copy(F.F), copy(F.piv))

Base.:(==)(F1::HemiCholesky, F2::HemiCholesky) = F1.L == F2.L
Base.:(==)(F1::HemiCholeskyReal, F2::HemiCholeskyReal) = F1.Lreal == F2.Lreal && F1.d == F2.d
Base.:(==)(F1::HemiCholeskyPivot, F2::HemiCholeskyPivot) = F1.F == F2.F && F1.piv == F2.piv

Base.isapprox(F1::HemiCholeskyReal, F2::HemiCholeskyReal; kwargs...) =
    isapprox(F1.Lreal, F2.Lreal; kwargs...) && F1.d == F2.d

function Base.getproperty(F::HemiCholesky, d::Symbol)
    d === :U && return F.L'
    return getfield(F, d)
end

Base.propertynames(F::HemiCholesky, private::Bool=false) =
    (:L, :U, (private ? fieldnames(typeof(F)) : ())...)

Base.size(F::HemiCholeskyReal) = size(F.Lreal)
Base.size(F::HemiCholeskyReal, d::Integer) = size(F.Lreal, d)

function Base.getproperty(F::HemiCholeskyReal{T}, d::Symbol) where T
    d === :L && return hrmatrix(T, F)
    d === :U && return hrmatrix(T, F)'
    return getfield(F, d)
end

Base.propertynames(F::HemiCholeskyReal, private::Bool=false) =
    (:L, :U, :d, (private ? fieldnames(typeof(F)) : ())...)

Base.propertynames(F::HemiCholeskyXY, private::Bool=false) =
    (:L, (private ? fieldnames(typeof(F)) : ())...)

function Base.getproperty(F::HemiCholeskyPivot{T}, d::Symbol) where T
    if d === :p
        return getfield(F, :piv)
    elseif d === :P
        n = size(F, 1)
        P = zeros(T, n, n)
        for i in 1:n
            P[getfield(F, :piv)[i], i] = one(T)
        end
        return P
    elseif d === :L
        return hrmatrix(T, F.F)
    elseif d === :U
        return hrmatrix(T, F.F)'
    else
        return getfield(F, d)
    end
end

Base.propertynames(F::HemiCholeskyPivot, private::Bool=false) =
    (:L, :U, :p, :P, (private ? (:piv,) : ())...)

# Base.eltype(::Type{HemiCholeskyXY{T,Ftype}}) where {T,Ftype} = PureHemi{T}

function _getL(F::HemiCholeskyReal{T}, i::Integer, j::Integer) where T
    d = F.d[j]
    nu = F.Lreal[i,j]
    ifelse(d == 0 && i==j, PureHemi{T}(1,0), PureHemi{T}(d*nu, nu))
end

function _getL(F::HemiCholesky{T}, i::Integer, j::Integer) where T
    i >= j ? F.L[i,j] : zero(PureHemi{T})
end
_getL(L::LowerTriangular{PureHemi{T}}, i::Integer, j::Integer) where T = L[i,j]

hrmatrix(::Type{T}, F::HemiCholesky) where T = convert(Matrix{PureHemi{T}}, F.L)
function hrmatrix(::Type{T}, F::HemiCholeskyReal) where T
    L = Matrix{PureHemi{T}}(undef, size(F))
    K = size(F, 1)
    for j = 1:K
        for i = 1:j-1
            L[i,j] = zero(PureHemi{T})
        end
        for i = j:K
            L[i,j] = _getL(F, i, j)
        end
    end
    return LowerTriangular(L)
end
hrmatrix(::Type{T}, F::HemiCholeskyPivot) where T = hrmatrix(T, F.F)
hrmatrix(::Type{T}, F::HemiCholeskyXY) where T = hrmatrix(T, F.F)

function Base.show(io::IO, ::MIME"text/plain", F::AbstractHemiCholesky)
    println(io, Base.dims2string(size(F)), " ", typeof(F), ':')
    _show(io, F)
end
_show(io::IO, F::AbstractHemiCholesky{T}) where T = Base.print_matrix(IOContext(io, :limit => true), hrmatrix(T, F))
function _show(io::IO, F::HemiCholeskyPivot{T}) where T
    Base.print_matrix(IOContext(io, :limit => true), hrmatrix(T, F))
    print(io, "\n  permutation: ", F.piv)
end
_show(io::IO, F::HemiCholeskyXY{T}) where T = _show(io, F.F)

LinearAlgebra.adjoint(F::AbstractHemiCholesky) = F

# Iteration for destructuring: L, U = cholesky(PureHemi, A)
# Not supported for HemiCholeskyPivot (consistent with CholeskyPivoted in LinearAlgebra).
Base.IteratorSize(::Type{<:HemiCholesky}) = Base.HasLength()
Base.IteratorSize(::Type{<:HemiCholeskyReal}) = Base.HasLength()
Base.length(::Union{HemiCholesky, HemiCholeskyReal}) = 2
Base.IteratorEltype(::Type{<:HemiCholesky}) = Base.EltypeUnknown()
Base.IteratorEltype(::Type{<:HemiCholeskyReal}) = Base.EltypeUnknown()
Base.iterate(F::HemiCholesky) = (F.L, Val(:U))
Base.iterate(F::HemiCholesky, ::Val{:U}) = (F.L', Val(:done))
Base.iterate(F::HemiCholeskyReal{T}) where T = (hrmatrix(T, F), Val(:U))
Base.iterate(F::HemiCholeskyReal{T}, ::Val{:U}) where T = (hrmatrix(T, F)', Val(:done))
Base.iterate(::AbstractHemiCholesky, ::Val{:done}) = nothing

function LinearAlgebra.AbstractMatrix(F::HemiCholeskyReal{T}) where T
    L_h = hrmatrix(T, F)
    return L_h * L_h'
end

function LinearAlgebra.AbstractMatrix(F::HemiCholesky{T}) where T
    return F.L * F.L'
end

function LinearAlgebra.AbstractMatrix(F::HemiCholeskyPivot{T}) where T
    M = AbstractMatrix(F.F)
    ip = invperm(F.piv)
    return M[ip, ip]
end

LinearAlgebra.AbstractMatrix(F::HemiCholeskyXY) = AbstractMatrix(F.F)

LinearAlgebra.AbstractArray(F::AbstractHemiCholesky) = AbstractMatrix(F)
Base.Matrix(F::AbstractHemiCholesky) = convert(Array, AbstractMatrix(F))
Base.Array(F::AbstractHemiCholesky) = Matrix(F)

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

"""
    cholesky(PureHemi{T}, A, pivot=NoPivot(); tol, blocksize) -> HemiCholeskyReal or HemiCholeskyPivot
    cholesky(PureHemi,    A, pivot=NoPivot(); tol, blocksize)

Compute the hemireal Cholesky factorization of the real symmetric matrix `A` and return a
[`HemiCholeskyReal`](@ref) (unpivoted) or [`HemiCholeskyPivot`](@ref) (pivoted) factorization
`F` satisfying `Matrix(F) ≈ A`.

Unlike the standard [`cholesky`](@ref LinearAlgebra.cholesky), hemireal Cholesky factorizations
exist for *all* real symmetric matrices, including indefinite and singular ones.

The type parameter `T` controls the working precision; use `PureHemi{T}` to fix it or `PureHemi`
to infer it from `eltype(A)`.

With `pivot = RowMaximum()`, diagonal pivoting is applied (maximizing the diagonal pivot at each
step) and the result is a [`HemiCholeskyPivot`](@ref) with the permutation accessible via `F.p`.

## Keyword arguments
- `tol`: diagonal entries (after partial elimination) with absolute value `≤ tol` are treated as
  zero. Defaults to a scale-adaptive multiple of machine epsilon.
- `blocksize`: block size for the cache-efficient blocked algorithm (dense matrices only).

See also [`cholesky!`](@ref), [`HemiCholeskyReal`](@ref), [`HemiCholeskyPivot`](@ref),
[`nullsolver`](@ref).
"""
function LinearAlgebra.cholesky(::Type{PureHemi{T}}, A::AbstractMatrix, pivot::Union{NoPivot,RowMaximum}=NoPivot(); tol=default_tol(A), blocksize=default_blocksize(T)) where T
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    A0 = Matrix{floattype(T)}(undef, size(A))
    copy!(A0, A)
    cholesky!(PureHemi{T}, A0, pivot; tol, blocksize=blocksize)
end
LinearAlgebra.cholesky(::Type{PureHemi}, A::AbstractMatrix, pivot::Union{NoPivot,RowMaximum}=NoPivot(); tol=default_tol(A), blocksize=default_blocksize(floattype(eltype(A)))) =
    cholesky(PureHemi{floattype(eltype(A))}, A, pivot; tol, blocksize=blocksize)

"""
    cholesky!(PureHemi{T}, A, pivot=NoPivot(); tol, blocksize) -> HemiCholeskyReal or HemiCholeskyPivot
    cholesky!(PureHemi,    A, pivot=NoPivot(); tol, blocksize)

The same as [`cholesky`](@ref), but overwrites the input matrix `A` with the lower-triangular
factor rather than allocating a copy.
"""
# Blocked, cache-friendly algorithm
function LinearAlgebra.cholesky!(::Type{PureHemi{T}}, A::AbstractMatrix{T}, ::NoPivot=NoPivot(); tol=default_tol(A), blocksize=default_blocksize(T)) where T
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
function LinearAlgebra.cholesky!(::Type{PureHemi{T}}, A::AbstractMatrix{T}, ::RowMaximum; tol=default_tol(A), blocksize=default_blocksize(T)) where T<:AbstractFloat
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


LinearAlgebra.cholesky!(::Type{PureHemi}, A::AbstractMatrix{T}, pivot::Union{NoPivot,RowMaximum}=NoPivot(); tol=default_tol(A), blocksize=default_blocksize(T)) where {T<:AbstractFloat} =
    cholesky!(PureHemi{T}, A, pivot; tol, blocksize=blocksize)


function solve_diagonal!(B, d, tol)
    K = size(B, 1)
    for j = 1:K
        Bjj = B[j,j]
        if abs(Bjj) > tol
            # compute ℓ (as the jth column of B)
            d[j] = Int8(sign(Bjj))
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
            d[j] = Int8(sign(Ajj))
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
    Cdt = (2 .* d) .* Ct
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

### Multiplication

function LinearAlgebra.mul!(out::AbstractVector{<:Real}, L::LowerTriangular{<:PureHemi}, y::AbstractVector{<:PureHemi})
    size(L, 2) == length(y) || throw(DimensionMismatch("length of y must match the number of columns of L"))
    size(L, 1) == length(out) || throw(DimensionMismatch("length of output must match the number of rows of L"))
    for i in 1:size(L, 1)
        s = zero(eltype(out))
        for j in 1:i
            s += L[i,j] * y[j]
        end
        out[i] = s
    end
    out
end

function LinearAlgebra.mul!(out::AbstractVector{<:PureHemi}, U::UpperTriangular{<:PureHemi}, x::AbstractVector{<:Real})
    size(U, 2) == length(x) || throw(DimensionMismatch("length of x must match the number of columns of U"))
    size(U, 1) == length(out) || throw(DimensionMismatch("length of output must match the number of rows of U"))
    for i in size(U, 1):-1:1
        s = zero(eltype(out))
        for j in i:size(U, 2)
            s += U[i,j] * x[j]
        end
        out[i] = s
    end
    out
end

function LinearAlgebra.mul!(out::AbstractMatrix{<:Real}, L::LowerTriangular{<:PureHemi}, U::UpperTriangular{<:PureHemi})
    size(out) == (size(L, 1), size(U, 2)) || throw(DimensionMismatch("output size must match the product of L and U"))
    size(L, 2) == size(U, 1) || throw(DimensionMismatch("inner dimensions of L and U must match"))
    for i in 1:size(L, 1)
        for j in 1:size(U, 2)
            s = zero(eltype(out))
            for k in 1:size(L, 2)
                s += L[i,k] * U[k,j]
            end
            out[i,j] = s
        end
    end
    out
end

### Solving linear systems

function solve_zeropivots(F; tol=default_tol(F))
    ns = nzerodiags(F)
    K = size(F, 1)
    T = real(eltype(F))
    X = Matrix{T}(undef, K, ns)
    H = Matrix{T}(undef, ns, ns)
    Y = Matrix{PureHemi{T}}(undef, K, ns)
    ns == 0 && return X, Y, lu!(H), Matrix{T}(undef, K, 0), falses(0)
    forwardsubst!(Y, F)
    backwardsubst!(X, H, F, Y)
    # Find the columns of X in the null space
    Hmax = maximum(abs, H; dims=2)
    nullflag = dropdims(Hmax; dims=2) .< tol
    # Find an orthonormal basis for the null space
    if sum(nullflag) == 0
        Q = Matrix{T}(undef, K, 0)   # this has no null space, they are all ordinary zero pivots
    else
        Q, _ = qr(X[:,nullflag])
    end
    # Prepare the solver for the non-null components of X
    HF = lu!(H[.!nullflag, .!nullflag])
    X, Y, HF, Q, nullflag
end
solve_zeropivots(F::HemiCholeskyPivot; tol=default_tol(F)) = solve_zeropivots(F.F; tol)

function LinearAlgebra.ldiv!(F::HemiCholeskyReal{T}, b::AbstractVector; forcenull::Bool=false) where T
    K = length(b)
    size(F, 1) == K || throw(DimensionMismatch("rhs must have length $K consistent with matrix size $(size(F,1))"))
    nnull = nzerodiags(F)
    nnull != 0 && !forcenull && error("There were zero diagonals; use `nullsolver(F)\\b` or, if you're sure all zeros correspond to null directions, `(\\)(F, b, forcenull=true)`.")
    ytilde = Vector{PureHemi{T}}(undef, K)
    forwardsubst!(ytilde, F, b)
    htilde = Vector{T}(undef, nnull)
    backwardsubst!(b, htilde, F, ytilde)
    return b
end

function LinearAlgebra.ldiv!(F::HemiCholeskyPivot{T}, b::AbstractVector; forcenull::Bool=false) where T
    size(F, 1) == length(b) || throw(DimensionMismatch("rhs must have length $(length(b)) consistent with matrix size $(size(F,1))"))
    permute!(b, F.piv)
    ldiv!(F.F, b; forcenull=forcenull)
    invpermute!(b, F.piv)
    return b
end

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::HemiCholeskyReal{T}; forcenull::Bool=false) where T
    m, n = size(B)
    size(F, 1) == n || throw(DimensionMismatch("matrix second dimension $n incompatible with factorization size $(size(F,1))"))
    b = Vector{T}(undef, n)
    for i in 1:m
        copyto!(b, view(B, i, :))
        ldiv!(F, b; forcenull=forcenull)
        copyto!(view(B, i, :), b)
    end
    return B
end

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::HemiCholeskyPivot{T}; forcenull::Bool=false) where T
    m, n = size(B)
    size(F, 1) == n || throw(DimensionMismatch("matrix second dimension $n incompatible with factorization size $(size(F,1))"))
    b = Vector{T}(undef, n)
    for i in 1:m
        copyto!(b, view(B, i, :))
        ldiv!(F, b; forcenull=forcenull)
        copyto!(view(B, i, :), b)
    end
    return B
end

function (\)(F::Union{HemiCholesky{T},HemiCholeskyReal{T},HemiCholeskyPivot{T}}, b::AbstractVector; forcenull::Bool=false) where T<:Real
    K = length(b)
    size(F,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(F)) of the matrix"))
    if F isa HemiCholesky
        bp, Lp = pivot(F, b)
        nnull = nzerodiags(Lp)
        if nnull != 0 && !forcenull
            error("There were zero diagonals; use `nullsolver(F)\\b` or, if you're sure all zeros correspond to null directions, (\\)(F, b, forcenull=true)`.")
        end
        ytilde = Vector{PureHemi{T}}(undef, K)
        forwardsubst!(ytilde, Lp, bp)
        xtilde = Vector{T}(undef, K)
        htilde = Vector{T}(undef, nnull)
        backwardsubst!(xtilde, htilde, Lp, ytilde)
        return ipivot(F, xtilde)
    else
        return ldiv!(F, Vector{T}(b); forcenull=forcenull)
    end
end

function (\)(F::HemiCholeskyXY{T}, b::AbstractVector) where T
    FF = F.F
    K = length(b)
    size(FF,1) == K || throw(DimensionMismatch("rhs must have a length ($(length(b))) consistent with the size $(size(FF)) of the matrix"))
    bp, Lp = pivot(FF, b)
    ytilde = Vector{PureHemi{T}}(undef, K)
    nnull = size(F.Q, 2)
    if nnull == 0
        forwardsubst!(ytilde, Lp, bp)
    else
        # project out the component of bp perpendicular to the null space
        bproj = F.Q'*bp
        forwardsubst!(ytilde, Lp, bp - F.Q*bproj)
    end
    ns = size(F.X, 2)
    htilde = Vector{T}(undef, ns)
    xtilde = Vector{T}(undef, K)
    backwardsubst!(xtilde, htilde, Lp, ytilde)
    ns == 0 && return ipivot(FF, xtilde)
    keep = .!F.nullflag
    α = -(F.HF\htilde[keep])
    x = xtilde+F.X[:,keep]*α
    # Return the least-squares answer
    if nnull > 0
        xproj = F.Q'*x
        x = x - F.Q*xproj
    end
    ipivot(FF, x)
end

# Forward-substitution with right hand side zero: find the
# symmetric-division nullspace of L
function forwardsubst!(Y, L::AbstractHemiCholesky)
    K, ns = size(Y, 1), size(Y, 2)
    T = real(eltype(Y))
    fill!(Y, zero(eltype(Y)))
    gα = Vector{T}(undef, ns)   # α-coefficient on current row
    si = 0              # number of zero-pivot columns processed so far
    for i = 1:K
        for jj = 1:si
            gα[jj] = 0
        end
        for j = 1:i-1
            Lij = _getL(L, i, j)
            for jj = 1:si
                gα[jj] -= Lij*Y[j,jj]
            end
        end
        Lii = _getL(L, i, i)
        if iszeropiv(Lii)
            for jj = 1:si
                Y[i,jj] = PureHemi{T}(0, -gα[jj]/Lii.m)
            end
            # Add a new zero-pivot column
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

function forwardsubst!(ytilde::AbstractVector, L::Union{AbstractHemiCholesky, LowerTriangular{<:PureHemi}}, b::AbstractVector)
    K = length(ytilde)
    length(b) == size(L, 1) == K || throw(DimensionMismatch("Sizes $(size(ytilde)), $(size(L)), and $(size(b)) do not match"))
    T = real(eltype(ytilde))
    for i = 1:K
        g = b[i]
        for j = 1:i-1
            Lij = _getL(L, i, j)
            g -= Lij*ytilde[j]
        end
        Lii = _getL(L, i, i)
        if iszeropiv(Lii)
            ytilde[i] = PureHemi{T}(0, -g/Lii.m)
        else
            ytilde[i] = g/Lii
        end
    end
    ytilde
end

function backwardsubst!(X, H, L::Union{AbstractHemiCholesky, LowerTriangular{<:PureHemi}}, Y)
    K, nc = size(Y, 1), size(Y, 2)
    size(X, 1) == K && size(X, 2) == nc || throw(DimensionMismatch("Sizes $(size(X)) and $(size(Y)) of X and Y must match"))
    T = real(eltype(Y))
    h = Vector{PureHemi{T}}(undef, nc)  # the current row
    si = ns = size(H, 1)        # number of zero pivots
    for i = K:-1:1
        for jj = 1:nc
            h[jj] = -Y[i,jj]   # - from conjugation (we're solving L'X = Y)
        end
        for j = i+1:K
            Lji = _getL(L, j, i)
            for jj = 1:nc
                h[jj] -= Lji*X[j,jj]
            end
        end
        Lii = _getL(L, i, i)
        if iszeropiv(Lii)
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
pivot(F::HemiCholeskyPivot, b) = b[F.piv], F.F

ipivot(L, b) = b
ipivot(F::HemiCholeskyPivot, b) = invpermute!(copy(b), F.piv)

iszeropiv(x::PureHemi) = x.n == 0

function nzerodiags(F::HemiCholesky)
    ns = 0
    for i = 1:size(F,1)
        ns += iszeropiv(F.L[i,i])
    end
    ns
end
function nzerodiags(F::HemiCholeskyReal)
    ns = 0
    for d in F.d
        ns += d == 0
    end
    ns
end
nzerodiags(F::HemiCholeskyPivot) = nzerodiags(F.F)
nzerodiags(F::HemiCholeskyXY) = size(F.X, 2)

floattype(::Type{T}) where {T<:AbstractFloat} = T
floattype(::Type{T}) where {T<:Integer} = Float64

const cachesize = 2^15

default_δ(A) = 10 * size(A, 1) * eps(floattype(real(eltype(A))))
default_tol(A) = default_δ(A) * maximum(abs, A)
function default_tol(F::HemiCholeskyReal)
    K = size(F, 1)
    Lreal = F.Lreal
    δ = default_δ(Lreal)
    K == 0 && return δ
    ma = zero(eltype(Lreal))
    for j = 1:K
        for i = j:K
            ma = max(ma, abs(Lreal[i,j]))
        end
    end
    δ * ma
end
default_tol(F::HemiCholeskyPivot) = default_tol(F.F)
default_blocksize(::Type{T}) where {T} = max(4, floor(Int, sqrt(cachesize/sizeof(T)/4)))

### Determinant

function LinearAlgebra.logabsdet(F::HemiCholeskyReal{T}) where T
    d = F.d
    n = size(F, 1)
    any(==(Int8(0)), d) && return (T(-Inf), one(T))
    sign_det = T(prod(Int.(d)))
    Lreal = F.Lreal
    logabs = n * log(2*one(T)) + 2 * sum(log(Lreal[j,j]) for j in 1:n)
    return (logabs, sign_det)
end

LinearAlgebra.logabsdet(F::Union{HemiCholeskyPivot,HemiCholeskyXY}) = logabsdet(F.F)

function LinearAlgebra.logdet(F::HemiCholeskyReal)
    logabs, sign = logabsdet(F)
    sign > 0 || throw(DomainError(F, "logdet requires a positive-definite matrix"))
    return logabs
end
LinearAlgebra.logdet(F::Union{HemiCholeskyPivot,HemiCholeskyXY}) = logdet(F.F)

function LinearAlgebra.det(F::AbstractHemiCholesky)
    logabs, sign = logabsdet(F)
    isinf(logabs) && return zero(real(eltype(F)))
    return sign * exp(logabs)
end
