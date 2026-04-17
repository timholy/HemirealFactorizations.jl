using HemirealNumbers, HemirealFactorizations
using LinearAlgebra
using SparseArrays
using Test
using Combinatorics
using DoubleFloats: Double64

@testset "Hemireal Cholesky factorization" begin

# ── Dense-only: exact output, pivoting variants, blocked algorithm, in-place ──

for p in (NoPivot(), RowMaximum())
    A = [2 0; 0 2]
    F = cholesky(PureHemi, A, p)
    @test Matrix(F) ≈ A
    @test rank(F) == 2
    A = [2 1; 1 2]
    F = cholesky(PureHemi, A, p)
    @test Matrix(F) ≈ A
    b = rand(size(A,2))
    x = F\b
    @test x ≈ A\b
end

# Dense-only: show output
let A = [0  1 -1;
         1  8 12;
        -1 12 20]
    F = cholesky(PureHemi, A)
    @test sprint(show, MIME("text/plain"), F) == "3×3 HemiCholeskyReal{Float64, Matrix{Float64}}:\n  1.0μ + 0.0ν       ⋅            ⋅     \n  0.0μ + 1.0ν  2.0μ + 2.0ν       ⋅     \n -0.0μ - 1.0ν  3.0μ + 3.0ν  1.0μ + 1.0ν"
    @test Matrix(F) ≈ [0 1 -1; 1 8 12; -1 12 20]
end

# Dense-only: blocked pivoting
let A = zeros(4,4)
    counter = 0
    for j = 1:4, i = j:4
        A[i,j] = A[j,i] = (counter+=1)
    end
    F = cholesky(PureHemi, A, RowMaximum())
    p = [4,1,2,3]
    @test F.piv == p
    @test Matrix(F.F) ≈ A[p,p]
    @test Matrix(F) ≈ A
    Fb = cholesky(PureHemi, A, RowMaximum(), blocksize=2)
    @test Fb.piv == p
    @test Matrix(F.F) ≈ Matrix(Fb.F)
    for pp in permutations([1,2,3,4])
        Fb = cholesky(PureHemi, A[pp,pp], RowMaximum(), blocksize=2)
        @test Fb.piv == permute!(invperm(pp), p)
        @test Matrix(F.F) ≈ Matrix(Fb.F)
        @test Matrix(Fb) ≈ A[pp,pp]
    end
end

# Dense-only: in-place cholesky! (tests blocked algorithm at size 200)
for p in (NoPivot(), RowMaximum())
    A = randn(201,200); A = A'*A
    F = cholesky!(PureHemi, copy(A), p)
    @test Matrix(F) ≈ A
    A[1,1] = 0
    F = cholesky!(PureHemi, copy(A), p)
    @test Matrix(F) ≈ A
    A = randn(199,200); A = A'*A
    F = cholesky!(PureHemi, copy(A), p)
    @test Matrix(F) ≈ A
end

# ── API features: dense and sparse ────────────────────────────────────────────

for (label, makeA) in [("dense", A -> A), ("sparse", A -> sparse(tril(A)))]

    # issuccess: always true regardless of definiteness
    @test issuccess(cholesky(PureHemi, makeA([2.0 1; 1 3])))
    @test issuccess(cholesky(PureHemi, makeA([-1.0 0; 0 1])))   # indefinite
    @test issuccess(cholesky(PureHemi, makeA([0.0 1; 1 0])))    # zero-pivot diagonal

    let A = [2.0 1; 1 3]
        # Iteration: L, U = F yields the lower- and upper-triangular PureHemi factors
        F = cholesky(PureHemi, makeA(A))
        L, U = F
        @test L isa Union{Matrix{<:PureHemi}, LowerTriangular{<:PureHemi}}
        @test L * L' ≈ A
        @test L * U ≈ A
        # .L and .U property and propertynames (:L, :U, :d)
        @test F.L == L
        @test F.U == U
        @test :L ∈ propertynames(F)
        @test :U ∈ propertynames(F)
        @test :d ∈ propertynames(F)
        # Direct forward and backward substitution with \
        b = [0.2, -0.7]
        y = L \ b
        @test eltype(y) === PureHemi{Float64}
        @test L * y ≈ b
        x = U \ y
        @test eltype(x) === Float64
        @test U * x ≈ y
        @test x ≈ A \ b
        @test L*U ≈ A
    end

    # rdiv!: (B / A) * A ≈ B
    let A = (X = rand(4, 4); X'*X + 4I)
        F = cholesky(PureHemi, makeA(A))
        B = rand(3, 4)
        R = rdiv!(copy(B), F)
        @test R * A ≈ B
    end

end

# ── API features: dense only (pivoting) ───────────────────────────────────────

# issuccess for pivoted factorization
@test issuccess(cholesky(PureHemi, [2.0 1; 1 3], RowMaximum()))

# getproperty / propertynames for HemiCholeskyPivot (.L, .U, .p, .P)
let A = [2.0 2 1; 2 3 1; 1 1 2]
    F = cholesky(PureHemi, A, RowMaximum())
    @test :L ∈ propertynames(F)
    @test :U ∈ propertynames(F)
    @test :p ∈ propertynames(F)
    @test :P ∈ propertynames(F)
    @test :piv ∉ propertynames(F)       # private, hidden by default
    @test F.p == F.piv                  # .p is an alias for .piv
    P = F.P
    @test P' * P ≈ I                    # P is a permutation matrix
    @test P * P' ≈ I
    @test Matrix(F.F) ≈ A[F.p, F.p]    # inner factor reconstructs permuted A
    L, U = F.F
    @test F.L == L                      # .L matches the inner HemiCholeskyReal's L
    @test F.U == U                      # .U matches the inner HemiCholeskyReal's U
    @test L * U ≈ A[F.p, F.p]
end

# rdiv! for HemiCholeskyPivot
let A = (X = rand(4, 4); X'*X + 4I)
    F = cholesky(PureHemi, A, RowMaximum())
    B = rand(3, 4)
    R = rdiv!(copy(B), F)
    @test R * A ≈ B
end

# show for HemiCholeskyPivot (includes "permutation:" line)
let A = [2.0 2 1; 2 3 1; 1 1 2]
    F = cholesky(PureHemi, A, RowMaximum())
    s = sprint(show, MIME("text/plain"), F)
    @test occursin("permutation:", s)
end

# ── HemiCholesky type (direct PureHemi matrix storage) ────────────────────────

# Positive-definite case: basic operations and backslash solve
let A = [2.0 1; 1 2]
    F_real = cholesky(PureHemi, A)
    L, U = F_real               # L is a Matrix{PureHemi{Float64}}
    F = HemiCholesky(L)

    @test size(F) == (2, 2)
    @test size(F, 1) == 2
    @test issuccess(F)
    @test isposdef(F)
    @test Matrix(F) ≈ A

    # Iteration
    L2, U2 = F
    @test L2 == L
    @test U2 == L'

    # Property access
    @test F.U == L'
    @test :L ∈ propertynames(F)
    @test :U ∈ propertynames(F)

    # copy and ==
    G = copy(F)
    @test G == F
    @test G !== F

    # show
    s = sprint(show, MIME("text/plain"), F)
    @test occursin("HemiCholesky", s)

    # Backslash solve (exercises the HemiCholesky branch in \)
    b = rand(2)
    @test F \ b ≈ A \ b
end

# Diagonal-zeros case: isposdef false and \ throws without nullsolver
let A = [0.0 1; 1 0]
    F_real = cholesky(PureHemi, A)
    L, U = F_real
    F = HemiCholesky(L)
    @test !isposdef(F)
    b = rand(2)
    @test_throws "There were zero diagonals" F \ b
    @test nullsolver(F_real) \ b ≈ A \ b
    Fpiv = cholesky(PureHemi, A, RowMaximum())
    @test_throws "There were zero diagonals" Fpiv \ b
    @test nullsolver(Fpiv) \ b ≈ A \ b
end

# show for HemiCholeskyXY
let A = (X = rand(3,5); X'*X)
    F = cholesky(PureHemi, A, tol=1e-10)
    Fs = nullsolver(F, tol=1e-10)
    s = sprint(show, MIME("text/plain"), Fs)
    @test occursin("HemiCholeskyReal", s)
end

# ── Correctness tests shared between dense and sparse ─────────────────────────
#
# makeA converts a dense symmetric Matrix to the type under test.
# For sparse we pass tril since the sparse factorization reads only i >= j.

@testset "$label" for (label, makeA) in [
    ("dense",  A -> A),
    ("sparse", A -> sparse(tril(A))),
]
    # Larger positive-definite matrix
    let A = (X = rand(7,5); X'*X)
        F = cholesky(PureHemi, makeA(A))
        @test Matrix(F) ≈ A
        b = rand(size(A, 2))
        @test F\b ≈ A\b
        @test rank(F) == 5
    end

    # Rank-deficient (singular) positive-semidefinite matrix
    let A = (X = rand(3,5); X'*X)
        F = cholesky(PureHemi, makeA(A), tol=1e-10)
        @test Matrix(F) ≈ A
        @test_throws ErrorException rank(F)
        Fs = nullsolver(F, tol=1e-10)
        @test rank(Fs) == 3
        b = rand(size(A, 2))
        @test nullsolver(F, tol=1e-10)\b ≈ svd(A)\b
    end

    # Indefinite matrix: diagonal entries of mixed sign
    let A = [-1.0 0; 0 1]
        F = cholesky(PureHemi, makeA(A))
        @test Matrix(F) ≈ A
        @test rank(F) == 2
        b = rand(size(A, 2))
        @test F\b ≈ A\b
    end

    # Matrix with zero diagonal (requires hemireal, not standard Cholesky)
    let A = [0.0 1; 1 0]
        F = cholesky(PureHemi, makeA(A))
        @test Matrix(F) ≈ A
        Fs = nullsolver(F)
        @test rank(Fs) == 2
        b = rand(size(A, 2))
        @test Fs\b ≈ A\b
    end

    # 3×3 matrix with zero diagonal entry that would normally need pivoting
    let A = [0.0  1 -1;
             1.0  8 12;
            -1.0 12 20]
        F = cholesky(PureHemi, makeA(A))
        @test Matrix(F) ≈ A
        Fs = nullsolver(F)
        @test rank(Fs) == 3
        b = rand(size(A, 2))
        @test Fs\b ≈ A\b
    end

    # Rank-2 matrix: sum of two outer products
    let a1 = [0.1, 0.2, 0.3], a2 = [-1.2, 0.8, 3.1]
        A = a1*a1' + a2*a2'
        b = rand(size(A, 2))
        xsvd = svd(A)\b
        F = cholesky(PureHemi, makeA(A), tol=1e-10)
        @test Matrix(F) ≈ A
        @test nullsolver(F, tol=1e-10)\b ≈ xsvd
    end

    # Singular indefinite matrix (one rank-1 outer product, zero diagonal)
    let a1 = [0.1, 0.2, 0.3]
        A = a1*a1'
        A[1,1] = 0.0
        b = rand(size(A, 1))
        xsvd = svd(A)\b
        F = cholesky(PureHemi, makeA(A))
        Fs = nullsolver(F)
        @test rank(Fs) == 2
        @test Fs\b ≈ xsvd
    end

    # isposdef: true only for positive-definite matrices
    let A = (X = rand(4, 4); X'*X + I)
        F = cholesky(PureHemi, makeA(A))
        @test isposdef(F)
        @test isposdef(nullsolver(F))
    end
    let A = [-1.0 0; 0 1]  # indefinite
        F = cholesky(PureHemi, makeA(A))
        @test !isposdef(F)
        @test !isposdef(nullsolver(F))
    end
    let A = [0.0 1; 1 0]   # also indefinite
        F = cholesky(PureHemi, makeA(A))
        @test !isposdef(F)
        @test !isposdef(nullsolver(F))
    end

    # copy: produces an independent equal copy
    let A = (X = rand(4, 4); X'*X)
        F = cholesky(PureHemi, makeA(A))
        G = copy(F)
        @test G == F
        @test G !== F
    end

    # ==: factorizations of equal matrices are equal; different matrices differ
    let A = (X = rand(4, 4); X'*X + I)
        F = cholesky(PureHemi, makeA(A))
        G = cholesky(PureHemi, makeA(A))
        @test F == G
        @test cholesky(PureHemi, makeA(A)) != cholesky(PureHemi, makeA(A + I))
        A1 = A + Diagonal(1e-10*rand(4))  # same sparsity pattern, different values
        G = cholesky(PureHemi, makeA(A1))
        @test F != G
        @test F ≈ G
    end
end # @testset "$label"

# ── Misc uncovered one-liners ─────────────────────────────────────────────────

# HemiCholeskyXY convenience constructor
let A = (X = rand(3,5); X'*X)
    F = cholesky(PureHemi, A, tol=1e-10)
    Fs1 = nullsolver(F, tol=1e-10)
    Fs2 = HemiCholeskyXY(F, tol=1e-10)   # convenience constructor with matching tol
    @test rank(Fs1) == rank(Fs2)
    @test Matrix(Fs1) ≈ Matrix(Fs2)
end

# isposdef, copy, == for HemiCholeskyPivot
let A = (X = rand(4, 4); X'*X + I)
    F = cholesky(PureHemi, A, RowMaximum())
    @test isposdef(F)
    G = copy(F)
    @test G == F
    @test G !== F
    @test cholesky(PureHemi, A, RowMaximum()) != cholesky(PureHemi, A + I, RowMaximum())
end

# propertynames for HemiCholeskyXY
let F = cholesky(PureHemi, [2.0 1; 1 2])
    Fs = nullsolver(F)
    @test :L ∈ propertynames(Fs)
end

# adjoint: F' === F
let F = cholesky(PureHemi, [2.0 1; 1 2])
    @test F' === F
end

# collect(F): exercises iterate(::AbstractHemiCholesky, ::Val{:done})
let A = [2.0 1; 1 2]
    F = cholesky(PureHemi, A)
    Lc, Uc = collect(F)
    @test Lc * Lc' ≈ A
end

# Array(F) and AbstractArray(F)
let A = [2.0 1; 1 2]
    F = cholesky(PureHemi, A)
    @test Array(F) ≈ A
end

# Matrix(Fs) for HemiCholeskyXY: exercises AbstractMatrix(F::HemiCholeskyXY)
let A = (X = rand(3,5); X'*X)
    F = cholesky(PureHemi, A, tol=1e-10)
    Fs = nullsolver(F, tol=1e-10)
    @test Matrix(Fs) ≈ A
end

# ── Determinant functions ─────────────────────────────────────────────────────

# Positive-definite: logabsdet/logdet/det for HemiCholeskyReal
let A = (X = rand(4, 4); X'*X + I)
    F = cholesky(PureHemi, A)
    la, s = logabsdet(F)
    @test s ≈ 1.0
    @test la ≈ log(det(A))
    @test logdet(F) ≈ log(det(A))
    @test det(F) ≈ det(A)
end

# Pivoted: delegates to inner HemiCholeskyReal
let A = (X = rand(4, 4); X'*X + I)
    F = cholesky(PureHemi, A, RowMaximum())
    @test det(F) ≈ det(A)
    @test logdet(F) ≈ log(det(A))
    @test logabsdet(F) == logabsdet(F.F)
end

# HemiCholeskyXY: delegates to inner factorization
let A = (X = rand(4, 4); X'*X + I)
    Fs = nullsolver(cholesky(PureHemi, A))
    @test det(Fs) ≈ det(A)
    @test logdet(Fs) ≈ log(det(A))
    @test logabsdet(Fs) == logabsdet(Fs.F)
end

# Singular: det = 0, logabsdet returns (-Inf, 1)
let A = (X = rand(3, 5); X'*X)   # 5×5, rank 3
    F = cholesky(PureHemi, A, tol=1e-10)
    la, s = logabsdet(F)
    @test la == -Inf
    @test s == 1.0
    @test det(F) == 0.0
end

# Indefinite (negative determinant): sign = -1, logdet throws DomainError
let A = [-1.0 0; 0 2]
    F = cholesky(PureHemi, A)
    la, s = logabsdet(F)
    @test s ≈ -1.0
    @test la ≈ log(abs(det(A)))
    @test det(F) ≈ det(A)
    @test_throws DomainError logdet(F)
    @test_throws DomainError logdet(cholesky(PureHemi, A, RowMaximum()))
    @test_throws DomainError logdet(nullsolver(cholesky(PureHemi, A)))
end

# cholesky! with PureHemi (no T) dispatch
let A = randn(4, 3); A = A'*A
    F = cholesky!(PureHemi, copy(A))
    @test Matrix(F) ≈ A
    F = cholesky!(PureHemi, copy(A), RowMaximum())
    @test Matrix(F) ≈ A
end

# ── Pure Julia fallbacks (non-BlasFloat via Double64) ─────────────────────────

@testset "Double64 (pure-Julia fallback path)" begin
    T = Double64
    for (label, makeA) in [("dense", A -> A), ("sparse", A -> sparse(tril(A)))]
        # Positive-definite
        let A = T[2 1; 1 2]
            F = cholesky(PureHemi{T}, makeA(A))
            @test Matrix(F) ≈ A
            b = rand(T, 2)
            @test F \ b ≈ A \ b
        end

        # Indefinite
        let A = T[-1 0; 0 1]
            F = cholesky(PureHemi{T}, makeA(A))
            @test Matrix(F) ≈ A
        end

        # Singular with nullsolver
        let X = T.(rand(5,3)); A_sym = X*X'   # rank-3, 5×5
            F = cholesky(PureHemi{T}, makeA(A_sym), tol=T(1e-20))
            @test Matrix(F) ≈ A_sym
        end
    end

    # Blocked algorithm (exercises update_columns! with vector d)
    let A = randn(20, 15); A = A'*A
        A64 = A
        AT = T.(A)
        F = cholesky!(PureHemi{T}, copy(AT), blocksize=4)
        @test Matrix(F) ≈ AT
    end

    # Pivoted
    let A = randn(5, 4); A = A'*A
        AT = T.(A)
        F = cholesky(PureHemi{T}, AT, RowMaximum())
        @test Matrix(F) ≈ AT
    end
end

end # @testset "Hemireal Cholesky factorization"
