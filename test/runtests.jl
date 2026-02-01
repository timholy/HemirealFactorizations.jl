using HemirealNumbers, HemirealFactorizations
using LinearAlgebra
using Test
using Combinatorics

@testset "Hemireal Cholesky factorization" begin
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

# larger positive-definite matrix
A = rand(7,5); A = A'*A
F = cholesky(PureHemi, A)
@test F*F' ≈ A
b = rand(size(A,2))
x = F\b
@test x ≈ A\b

# singular matrix
A = rand(3,5); A = A'*A
F = cholesky(PureHemi, A, tol=1e-10)
@test F*F' ≈ A
@test_throws ErrorException rank(F) == 3
@test rank(nullsolver(F, tol=1e-10)) == 3
# add pivoting
Fp = cholesky(PureHemi, A, Val{true}, tol=1e-10)
@test rank(nullsolver(Fp, tol=1e-10)) == 3

# An indefinite matrix
A = [-1 0; 0 1]
F = cholesky(PureHemi, A)
@test F*F' ≈ A
@test rank(F) == 2
b = rand(size(A,2))
x = F\b
@test x ≈ A\b
# A matrix that has no conventional LL' or LDL' factorization
A = [0 1; 1 0]
F = cholesky(PureHemi, A)
@test F*F' ≈ A
Fs = nullsolver(F)
@test rank(Fs) == 2
b = rand(size(A,2))
x = Fs\b
@test x ≈ A\b

# A matrix that would normally require pivoting
A = [0  1 -1;
     1  8 12;
    -1 12 20]
F = cholesky(PureHemi, A)
@test sprint(show, MIME("text/plain"), F) == """
3×3 HemirealFactorizations.HemiCholeskyReal{Float64}:
  1.0μ + 0.0ν  0.0μ + 0.0ν  0.0μ + 0.0ν
  0.0μ + 1.0ν  2.0μ + 2.0ν  0.0μ + 0.0ν
 -0.0μ - 1.0ν  3.0μ + 3.0ν  1.0μ + 1.0ν"""
@test F*F' ≈ A
Fs = nullsolver(F)
@test rank(Fs) == 3
@test convert(Matrix, F) == PureHemi{Float64}[μ 0 0; ν 2μ+2ν 0; -ν 3μ+3ν μ+ν]
b = rand(size(A,2))
x = Fs\b
@test x ≈ A\b
# also try pivoting
Fp = cholesky(PureHemi, A, Val{true})
@test x ≈ Fp\b

# Blocked pivoting
A = zeros(4,4)
counter = 0
for j = 1:4
    for i = j:4
        A[i,j] = A[j,i] = (counter+=1)
    end
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

# A singular matrix
a1 = [0.1,0.2,0.3]
a2 = [-1.2,0.8,3.1]
A = a1*a1' + a2*a2'
b = rand(size(A,2))
xsvd = svd(A)\b
F = cholesky(PureHemi, A, tol=1e-10)
@test F*F' ≈ A
x = nullsolver(F)\b
@test x ≈ xsvd
Fp = cholesky(PureHemi, A, Val{true}, tol=1e-10)
xp = nullsolver(Fp)\b
@test xp ≈ xsvd

# A singular indefinite matrix
A = a1*a1'
A[1,1] = 0
F = cholesky(PureHemi, A)
Fs = nullsolver(F)
@test rank(Fs) == 2
@test Fs.nullflag == [false,true]
b = rand(size(F, 1))
xsvd = svd(A)\b
x = Fs\b
@test x ≈ xsvd
Fp = cholesky(PureHemi, A, Val{true}, tol=1e-10)
xp = nullsolver(Fp)\b
@test xp ≈ xsvd

# In-place versions. Make these big enough to test blocked algorithm.
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
end
