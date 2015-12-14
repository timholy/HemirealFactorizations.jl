using HemirealNumbers, HemirealFactorizations
using Base.Test

const mult_safe = VERSION >= v"0.5.0-dev"   # julia #14293

A = [2 0; 0 2]
F = cholfact(PureHemi, A)
@test convert(Matrix, F) == [μ+ν 0; 0 μ+ν]
@test rank(F) == 2
A = [2 1; 1 2]
F = cholfact(PureHemi, A)
@test convert(Matrix, F) == [μ+ν 0; (μ+ν)/2 (sqrt(3)/2)*(μ+ν)]
@test_approx_eq F*F' A
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

A = rand(7,5); A = A'*A
F = cholfact(PureHemi, A)
if mult_safe
    @test_approx_eq F*F' A
end
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

A = rand(3,5); A = A'*A
F = cholfact(PureHemi, A, tol=1e-10)
if mult_safe
    @test_approx_eq F*F' A
end
@test_throws ErrorException rank(F) == 3
@test rank(nullsolver(F, tol=1e-10)) == 3

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

# In-place versions. Make these big enough to test blocked algorithm.
A = randn(201,200); A = A'*A
F = cholfact!(PureHemi, copy(A))
if mult_safe
    @test_approx_eq F*F' A
end
A[1,1] = 0
F = cholfact!(PureHemi, copy(A))
if mult_safe
    @test_approx_eq F*F' A
end
A = randn(199,200); A = A'*A
F = cholfact!(PureHemi, copy(A))
if mult_safe
    @test_approx_eq F*F' A
end
