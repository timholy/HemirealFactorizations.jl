using HemirealNumbers, HemirealFactorizations
using LinearAlgebra
using SparseArrays
using Test
using Combinatorics

@testset "Hemireal Cholesky factorization" begin

# ── Dense-only: exact output, pivoting variants, blocked algorithm, in-place ──

for p in (Val{false}, Val{true})
    A = [2 0; 0 2]
    F = cholesky(PureHemi, A, p)
    @test convert(Matrix, F) == [μ+ν 0; 0 μ+ν]
    @test rank(F) == 2
    A = [2 1; 1 2]
    F = cholesky(PureHemi, A, p)
    @test convert(Matrix, F) == [μ+ν 0; (μ+ν)/2 (sqrt(3)/2)*(μ+ν)]
    @test F*F' ≈ A
    b = rand(size(A,2))
    x = F\b
    @test x ≈ A\b
end

# Dense-only: show output
let A = [0  1 -1;
         1  8 12;
        -1 12 20]
    F = cholesky(PureHemi, A)
    @test sprint(show, MIME("text/plain"), F) == """
3×3 HemirealFactorizations.HemiCholeskyReal{Float64}:
  1.0μ + 0.0ν  0.0μ + 0.0ν  0.0μ + 0.0ν
  0.0μ + 1.0ν  2.0μ + 2.0ν  0.0μ + 0.0ν
 -0.0μ - 1.0ν  3.0μ + 3.0ν  1.0μ + 1.0ν"""
    @test convert(Matrix, F) == PureHemi{Float64}[μ 0 0; ν 2μ+2ν 0; -ν 3μ+3ν μ+ν]
end

# Dense-only: blocked pivoting
let A = zeros(4,4)
    counter = 0
    for j = 1:4, i = j:4
        A[i,j] = A[j,i] = (counter+=1)
    end
    F = cholesky(PureHemi, A, Val{true})
    p = [4,1,2,3]
    @test F.piv == p
    @test F.L*F.L' ≈ A[p,p]
    @test F*F' ≈ A
    Fb = cholesky(PureHemi, A, Val{true}, blocksize=2)
    @test Fb.piv == p
    @test F.L.L ≈ Fb.L.L
    for pp in permutations([1,2,3,4])
        Fb = cholesky(PureHemi, A[pp,pp], Val{true}, blocksize=2)
        @test Fb.piv == permute!(invperm(pp), p)
        @test LowerTriangular(F.L.L) ≈ LowerTriangular(Fb.L.L)
        @test Fb*Fb' ≈ A[pp,pp]
    end
end

# Dense-only: in-place cholesky! (tests blocked algorithm at size 200)
for p in (Val{false}, Val{true})
    A = randn(201,200); A = A'*A
    F = cholesky!(PureHemi, copy(A), p)
    @test F*F' ≈ A
    A[1,1] = 0
    F = cholesky!(PureHemi, copy(A), p)
    @test F*F' ≈ A
    A = randn(199,200); A = A'*A
    F = cholesky!(PureHemi, copy(A), p)
    @test F*F' ≈ A
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
        @test F*F' ≈ A
        b = rand(size(A, 2))
        @test F\b ≈ A\b
        @test rank(F) == 5
    end

    # Rank-deficient (singular) positive-semidefinite matrix
    let A = (X = rand(3,5); X'*X)
        F = cholesky(PureHemi, makeA(A), tol=1e-10)
        @test F*F' ≈ A
        @test_throws ErrorException rank(F)
        Fs = nullsolver(F, tol=1e-10)
        @test rank(Fs) == 3
        b = rand(size(A, 2))
        @test nullsolver(F, tol=1e-10)\b ≈ svd(A)\b
    end

    # Indefinite matrix: diagonal entries of mixed sign
    let A = [-1.0 0; 0 1]
        F = cholesky(PureHemi, makeA(A))
        @test F*F' ≈ A
        @test rank(F) == 2
        b = rand(size(A, 2))
        @test F\b ≈ A\b
    end

    # Matrix with zero diagonal (requires hemireal, not standard Cholesky)
    let A = [0.0 1; 1 0]
        F = cholesky(PureHemi, makeA(A))
        @test F*F' ≈ A
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
        @test F*F' ≈ A
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
        @test F*F' ≈ A
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
end # @testset "$label"

end # @testset "Hemireal Cholesky factorization"
