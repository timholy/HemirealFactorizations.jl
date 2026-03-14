import SparseArrays: SparseMatrixCSC, nzrange, rowvals, nonzeros

### Sparse factorization type

struct SparseHemiCholeskyReal{T<:Real} <: AbstractHemiCholesky{T}
    L::SparseMatrixCSC{T,Int}
    d::Vector{Int8}
end

Base.size(F::SparseHemiCholeskyReal) = size(F.L)
Base.size(F::SparseHemiCholeskyReal, dim::Integer) = size(F.L, dim)
Base.eltype(::Type{SparseHemiCholeskyReal{T}}) where T = PureHemi{T}

function _getL(F::SparseHemiCholeskyReal{T}, i::Integer, j::Integer) where T
    dj = F.d[j]
    nu = F.L[i,j]
    ifelse(dj == 0 && i == j, PureHemi{T}(1, 0), PureHemi{T}(dj * nu, nu))
end

Base.copy(F::SparseHemiCholeskyReal) = SparseHemiCholeskyReal(copy(F.L), copy(F.d))
Base.:(==)(F1::SparseHemiCholeskyReal, F2::SparseHemiCholeskyReal) = F1.L == F2.L && F1.d == F2.d
LinearAlgebra.isposdef(F::SparseHemiCholeskyReal) = all(==(Int8(1)), F.d)

function hrmatrix(::Type{T}, F::SparseHemiCholeskyReal) where T
    n = size(F, 1)
    L = Array{PureHemi{T}}(undef, n, n)
    for j = 1:n
        for i = 1:j-1
            L[i,j] = zero(PureHemi{T})
        end
        for i = j:n
            L[i,j] = _getL(F, i, j)
        end
    end
    L
end

function nzerodiags(F::SparseHemiCholeskyReal)
    count(==(0), F.d)
end

function default_tol(A::SparseMatrixCSC)
    δ = default_δ(A)
    nz = nonzeros(A)
    isempty(nz) && return δ
    δ * maximum(abs, nz)
end

function default_tol(F::SparseHemiCholeskyReal)
    δ = default_δ(F.L)
    nz = nonzeros(F.L)
    isempty(nz) && return δ
    δ * maximum(abs, nz)
end

### Factorization

# Left-looking sparse hemi-Cholesky factorization.
#
# The algorithm processes columns j = 1:n in order.  Before computing column j
# of L it applies all pending updates from the already-factored columns k < j
# (the "left-looking" style).  A single dense scatter vector w[1:n] is used to
# assemble each column; updates are applied by iterating only over the structural
# nonzeros of the already-computed columns, keeping the update step O(nnz) rather
# than O(n).
#
# A is assumed to store the lower triangle of a symmetric matrix (entries with
# row >= col).  Pass tril(A) explicitly if A contains both triangles.

function LinearAlgebra.cholesky(::Type{PureHemi{T}}, A::SparseMatrixCSC; tol=default_tol(A)) where T<:AbstractFloat
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    n = size(A, 1)

    # Dense scatter workspace: w[i] accumulates the current column being assembled.
    # w_flag[i] tracks whether position i has been written (avoids scanning all n
    # entries when resetting).  w_nnz collects the written indices.
    w      = zeros(T, n)
    w_flag = fill(false, n)
    w_nnz  = sizehint!(Int[], n)          # indices written to w this column (unsorted until needed)

    # L is built column by column.  L_rows[j] and L_vals[j] are stored in
    # strictly increasing row order (diagonal first), which lets us use
    # searchsortedfirst when the left-looking update needs L[j,k] for a specific j.
    L_rows = [empty!(Vector{Int}(undef, n)) for _ in 1:n]
    L_vals = [empty!(Vector{T}(undef, n)) for _ in 1:n]

    d = Vector{Int8}(undef, n)

    # row_preds[i] = list of k < i (in increasing order of k) where L[i,k] ≠ 0.
    # When processing column i we iterate over row_preds[i] to collect updates.
    row_preds = [empty!(Vector{Int}(undef, n)) for _ in 1:n]

    for j in 1:n
        # ── Step 1: scatter lower-triangle of A[:,j] into w ──────────────────
        for idx in nzrange(A, j)
            i = rowvals(A)[idx]
            i >= j || continue
            mark!(w, w_flag, w_nnz, i, T(nonzeros(A)[idx]))
        end

        # ── Step 2: left-looking update from columns k < j ───────────────────
        # For each k < j with L[j,k] ≠ 0, subtract 2·d[k]·L[j,k]·L[i,k] from
        # w[i] for every i ≥ j that has a nonzero in column k of L.
        for k in row_preds[j]
            dk = d[k]
            dk == 0 && continue          # singular column: contribution is zero

            # Locate row j inside the sorted column k of L (guaranteed to exist).
            pos = searchsortedfirst(L_rows[k], j)
            Ljk = L_vals[k][pos]
            c   = 2 * dk * Ljk           # scalar that multiplies each L[i,k]

            for ptr in pos:length(L_rows[k])
                i = L_rows[k][ptr]
                mark!(w, w_flag, w_nnz, i, -c * L_vals[k][ptr])
            end
        end

        # ── Step 3: compute diagonal and off-diagonal entries of L[:,j] ──────
        Ajj = w[j]

        # Sort now so L_rows[j] is built in increasing row order (required by
        # SparseMatrixCSC and by the searchsortedfirst calls in the left-looking update).
        sort!(w_nnz)

        if abs(Ajj) <= tol
            d[j] = 0
            # For a singular column the rank-1 update to future columns is zero
            # (factor 2*d[j] = 0), so we do NOT add j to row_preds[i].
            # However, the off-diagonal entries are still needed by forward/backward
            # substitution: _getL returns PureHemi(0, L[i,j]) for d[j]=0, i>j.
            for i in w_nnz
                i <= j && continue
                vi = w[i]
                iszero(vi) && continue
                push!(L_rows[j], i)
                push!(L_vals[j], vi)
            end
        else
            dj  = Int8(sign(Ajj))
            d[j] = dj
            Ljj  = sqrt(abs(Ajj) / 2)      # ν component of the diagonal entry
            f    = dj / (2 * Ljj)           # = sign(Ajj)/sqrt(2|Ajj|)

            push!(L_rows[j], j)
            push!(L_vals[j], Ljj)

            for i in w_nnz
                i <= j && continue
                Lij = w[i] * f
                iszero(Lij) && continue
                push!(L_rows[j], i)
                push!(L_vals[j], Lij)
                # row_preds[i] accumulates j values in increasing order since j
                # strictly increases across iterations.
                push!(row_preds[i], j)
            end
        end

        # ── Step 4: reset workspace ───────────────────────────────────────────
        for i in w_nnz
            w[i]      = zero(T)
            w_flag[i] = false
        end
        empty!(w_nnz)
    end

    # ── Assemble SparseMatrixCSC for L ────────────────────────────────────────
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    for j in 1:n
        colptr[j+1] = colptr[j] + length(L_rows[j])
    end
    total_nnz = colptr[n + 1] - 1

    rowval = Vector{Int}(undef, total_nnz)
    nzval  = Vector{T}(undef,  total_nnz)
    for j in 1:n
        r = colptr[j]:colptr[j+1]-1
        isempty(r) && continue
        copyto!(rowval, first(r), L_rows[j], 1, length(r))
        copyto!(nzval,  first(r), L_vals[j], 1, length(r))
    end

    L = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    return SparseHemiCholeskyReal{T}(L, d)
end

LinearAlgebra.cholesky(::Type{PureHemi}, A::SparseMatrixCSC; kwargs...) =
    cholesky(PureHemi{floattype(eltype(A))}, A; kwargs...)

### Solve and nullspace

function nullsolver(F::SparseHemiCholeskyReal; tol=default_tol(F))
    X, Y, HF, Q, nullflag = solve_singularities(F; tol=tol)
    HemiCholeskyXY{eltype(F.L), typeof(F), typeof(HF)}(F, X, Y, HF, Q, nullflag)
end

function LinearAlgebra.AbstractMatrix(F::SparseHemiCholeskyReal{T}) where T
    L_h = hrmatrix(T, F)
    return L_h * L_h'
end

function LinearAlgebra.ldiv!(F::SparseHemiCholeskyReal{T}, b::AbstractVector; forcenull::Bool=false) where T
    K = length(b)
    size(F, 1) == K || throw(DimensionMismatch("rhs must have length $K consistent with matrix size $(size(F,1))"))
    nnull = nzerodiags(F)
    nnull != 0 && !forcenull && error("There were zero diagonals; use `nullsolver(F)\\b` or pass `forcenull=true`.")
    ytilde = Vector{PureHemi{T}}(undef, K)
    forwardsubst!(ytilde, F, b)
    xtilde = Vector{T}(undef, K)
    htilde = Vector{T}(undef, nnull)
    backwardsubst!(xtilde, htilde, F, ytilde)
    copyto!(b, xtilde)
    return b
end

function Base.:(\)(F::SparseHemiCholeskyReal{T}, b::AbstractVector; forcenull::Bool=false) where T<:Real
    K = length(b)
    size(F, 1) == K || throw(DimensionMismatch("rhs length $K does not match matrix size $(size(F,1))"))
    return ldiv!(F, Vector{T}(b); forcenull=forcenull)
end

@inline function mark!(w, w_flag, w_nnz, i, v)
    if !w_flag[i]
        w_flag[i] = true
        push!(w_nnz, i)
    end
    w[i] += v
end
