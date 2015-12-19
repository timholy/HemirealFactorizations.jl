using HemirealNumbers, HemirealFactorizations
using Base.Test
if VERSION >= v"0.5.0-dev"
    using Combinatorics
end

const mult_safe = VERSION >= v"0.5.0-dev"   # julia #14293

for p in (Val{false}, Val{true})
    A = [2 0; 0 2]
    F = cholfact(PureHemi, A, p)
    @test convert(Matrix, F) == [μ+ν 0; 0 μ+ν]
    @test rank(F) == 2
    A = [2 1; 1 2]
    F = cholfact(PureHemi, A, p)
    @test convert(Matrix, F) == [μ+ν 0; (μ+ν)/2 (sqrt(3)/2)*(μ+ν)]
    @test_approx_eq F*F' A
    b = rand(size(A,2))
    x = F\b
    @test_approx_eq x A\b
end

# larger positive-definite matrix
A = rand(7,5); A = A'*A
F = cholfact(PureHemi, A)
if mult_safe
    @test_approx_eq F*F' A
end
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

# singular matrix
A = rand(3,5); A = A'*A
F = cholfact(PureHemi, A, tol=1e-10)
if mult_safe
    @test_approx_eq F*F' A
end
@test_throws ErrorException rank(F) == 3
@test rank(nullsolver(F, tol=1e-10)) == 3
# add pivoting
Fp = cholfact(PureHemi, A, Val{true}, tol=1e-10)
@test rank(nullsolver(Fp, tol=1e-10)) == 3

# An indefinite matrix
A = [-1 0; 0 1]
F = cholfact(PureHemi, A)
@test_approx_eq F*F' A
@test rank(F) == 2
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

# A matrix that has no conventional LL' or LDL' factorization
A = [0 1; 1 0]
F = cholfact(PureHemi, A)
@test_approx_eq F*F' A
Fs = nullsolver(F)
@test rank(Fs) == 2
b = rand(size(A,2))
x = Fs\b
@test_approx_eq x A\b

# A matrix that would normally require pivoting
A = [0  1 -1;
     1  8 12;
    -1 12 20]
F = cholfact(PureHemi, A)
@test_approx_eq F*F' A
Fs = nullsolver(F)
@test rank(Fs) == 3
@test_approx_eq convert(Matrix, F) [μ 0 0; ν 2μ+2ν 0; -ν 3μ+3ν μ+ν]
b = rand(size(A,2))
x = Fs\b
@test_approx_eq x A\b
# also try pivoting
Fp = cholfact(PureHemi, A, Val{true})
@test_approx_eq x Fp\b

# Blocked pivoting
A = zeros(4,4)
counter = 0
for j = 1:4
    for i = j:4
        A[i,j] = A[j,i] = (counter+=1)
    end
end
F = cholfact(PureHemi, A, Val{true})
p = [4,1,2,3]
@test F.piv == p
@test_approx_eq F.L*F.L' A[p,p]
@test_approx_eq F*F' A
Fb = cholfact(PureHemi, A, Val{true}, blocksize=2)
@test Fb.piv == p
@test_approx_eq F.L.L Fb.L.L
for pp in permutations([1,2,3,4])
    Fb = cholfact(PureHemi, A[pp,pp], Val{true}, blocksize=2)
    @test Fb.piv == permute!(invperm(pp), p)
    @test_approx_eq LowerTriangular(F.L.L) LowerTriangular(Fb.L.L)
    @test_approx_eq Fb*Fb' A[pp,pp]
end

# A singular matrix
a1 = [0.1,0.2,0.3]
a2 = [-1.2,0.8,3.1]
A = a1*a1' + a2*a2'
b = rand(size(A,2))
xsvd = svdfact(A)\b
F = cholfact(PureHemi, A, tol=1e-10)
@test_approx_eq F*F' A
x = nullsolver(F)\b
@test_approx_eq_eps x xsvd 1e-10
Fp = cholfact(PureHemi, A, Val{true}, tol=1e-10)
xp = nullsolver(Fp)\b
@test_approx_eq_eps xp xsvd 1e-10

# A singular indefinite matrix
A = a1*a1'
A[1,1] = 0
F = cholfact(PureHemi, A)
Fs = nullsolver(F)
@test rank(Fs) == 2
@test Fs.nullflag == [false,true]
b = rand(size(F, 1))
xsvd = svdfact(A)\b
x = Fs\b
@test_approx_eq_eps x xsvd 1e-10
Fp = cholfact(PureHemi, A, Val{true}, tol=1e-10)
xp = nullsolver(Fp)\b
@test_approx_eq_eps xp xsvd 1e-10

# In-place versions. Make these big enough to test blocked algorithm.
for p in (Val{false}, Val{true})
    A = randn(201,200); A = A'*A
    F = cholfact!(PureHemi, copy(A), p)
    if mult_safe
        @test_approx_eq F*F' A
    end
    A[1,1] = 0
    F = cholfact!(PureHemi, copy(A), p)
    if mult_safe
        @test_approx_eq F*F' A
    end
    A = randn(199,200); A = A'*A
    F = cholfact!(PureHemi, copy(A), p)
    if mult_safe
        @test_approx_eq F*F' A
    end
end
