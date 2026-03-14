using HemirealNumbers, HemirealFactorizations
using LinearAlgebra
using SparseArrays
using Test
using Combinatorics

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
    @test sprint(show, MIME("text/plain"), F) == "3×3 HemiCholeskyReal{Float64, Matrix{Float64}}:\n  1.0μ + 0.0ν  0.0μ + 0.0ν  0.0μ + 0.0ν\n  0.0μ + 1.0ν  2.0μ + 2.0ν  0.0μ + 0.0ν\n -0.0μ - 1.0ν  3.0μ + 3.0ν  1.0μ + 1.0ν"
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
    @test Matrix(F.L) ≈ A[p,p]
    @test Matrix(F) ≈ A
    Fb = cholesky(PureHemi, A, RowMaximum(), blocksize=2)
    @test Fb.piv == p
    @test F.L.L ≈ Fb.L.L
    for pp in permutations([1,2,3,4])
        Fb = cholesky(PureHemi, A[pp,pp], RowMaximum(), blocksize=2)
        @test Fb.piv == permute!(invperm(pp), p)
        @test LowerTriangular(F.L.L) ≈ LowerTriangular(Fb.L.L)
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

# ── Dense-only: new API features ──────────────────────────────────────────────

# issuccess: always true regardless of definiteness
@test issuccess(cholesky(PureHemi, [2.0 1; 1 3]))
@test issuccess(cholesky(PureHemi, [-1.0 0; 0 1]))   # indefinite
@test issuccess(cholesky(PureHemi, [0.0 1; 1 0]))    # singular diagonal
@test issuccess(cholesky(PureHemi, [2.0 1; 1 3], RowMaximum()))

# Iteration: destructure F to obtain the PureHemi lower-triangular factor L
let A = [2.0 1; 1 3]
    F = cholesky(PureHemi, A)
    L, U = F
    @test L isa Matrix{<:PureHemi}
    @test L * L' ≈ A
    @test L * U ≈ A
end

# .U property: upper-triangular factor (adjoint of L), available on all types
let A = [2.0 1; 1 3]
    F = cholesky(PureHemi, A)
    L, U = F
    @test F.U == U                     # .U matches second element of iteration
    @test L * F.U ≈ A
end
let A = [2.0 2 1; 2 3 1; 1 1 2]
    F = cholesky(PureHemi, A, RowMaximum())
    L, U = F.L                         # unpack inner HemiCholeskyReal
    @test F.U == U                     # HemiCholeskyPivot.U matches inner factor's U
    @test L * F.U ≈ A[F.p, F.p]
end

# getproperty / propertynames for HemiCholeskyPivot
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
    @test Matrix(F.L) ≈ A[F.p, F.p]    # inner factor reconstructs permuted A
end

# propertynames for HemiCholeskyReal
let F = cholesky(PureHemi, [2.0 1; 1 3])
    @test :L ∈ propertynames(F)
    @test :U ∈ propertynames(F)
    @test :d ∈ propertynames(F)
end

# rdiv!: rdiv!(copy(B), F) computes B / A, verified by (B / A) * A ≈ B
for p in (NoPivot(), RowMaximum())
    A = (X = rand(4, 4); X'*X + 4I)    # positive-definite, ensures full rank
    F = cholesky(PureHemi, A, p)
    B = rand(3, 4)
    R = rdiv!(copy(B), F)
    @test R * A ≈ B
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
        @test isposdef(cholesky(PureHemi, makeA(A)))
    end
    let A = [-1.0 0; 0 1]
        @test !isposdef(cholesky(PureHemi, makeA(A)))  # indefinite
    end
    let A = [0.0 1; 1 0]
        @test !isposdef(cholesky(PureHemi, makeA(A)))  # zero diagonal → singular
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
    end
end # @testset "$label"

end # @testset "Hemireal Cholesky factorization"
