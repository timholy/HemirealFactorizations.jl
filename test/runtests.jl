using HemirealNumbers, HemirealFactorizations
using Base.Test

const mult_safe = VERSION >= v"0.5.0-dev"   # julia #14293

A = [2 0; 0 2]
F = cholfact(PureHemi, A)
@test convert(Matrix, F) == [μ+ν 0; 0 μ+ν]
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
F = cholfact(PureHemi, A)
if mult_safe
    @test_approx_eq F*F' A
end

# An indefinite matrix
A = [-1 0; 0 1]
F = cholfact(PureHemi, A)
@test_approx_eq F*F' A
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

# A matrix that has no conventional LL' or LDL' factorization
A = [0 1; 1 0]
F = cholfact(PureHemi, A)
@test_approx_eq F*F' A
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

# A matrix that would normally require pivoting
A = [0  1 -1;
     1  8 12;
    -1 12 20]
F = cholfact(PureHemi, A)
@test_approx_eq F*F' A
@test_approx_eq convert(Matrix, F) [μ 0 0; ν 2μ+2ν 0; -ν 3μ+3ν μ+ν]
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

# A singular matrix
A = rand(2,3); A = A'*A
b = rand(size(A,2))
xsvd = svdfact(A)\b
bsvd = A*xsvd
F = cholfact(PureHemi, A)
@test_approx_eq F*F' A
x = F\bsvd
@test_approx_eq_eps A*x bsvd 1e-10

# rhs not within the range of the matrix
x = F\b
bchol = A*x
Δb = bchol - b
# Test that the correction is limited to the 3rd entry
@test abs(Δb[1]) < 1e-12
@test abs(Δb[2]) < 1e-12

# In-place versions. Make these big enough to test blocked algorithm.
A = randn(201,200); A = A'*A
F = cholfact!(PureHemi, copy(A))
@test_approx_eq F*F' A
A[1,1] = 0
F = cholfact!(PureHemi, copy(A))
@test_approx_eq F*F' A
A = randn(199,200); A = A'*A
F = cholfact!(PureHemi, copy(A))
@test_approx_eq F*F' A
