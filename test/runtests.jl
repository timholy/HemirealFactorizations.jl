using HemirealNumbers, HemirealFactorizations
using Base.Test

const mult_safe = VERSION >= v"0.5.0-dev"   # julia #14293

A = [2 0; 0 2]
F = cholfact(PureHemi, A)
@test F.L == [μ+ν 0; 0 μ+ν]
A = [2 1; 1 2]
F = cholfact(PureHemi, A)
@test F.L == [μ+ν 0; (μ+ν)/2 (sqrt(3)/2)*(μ+ν)]
@test_approx_eq F.L*F.L' A
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

A = rand(7,5); A = A'*A
F = cholfact(PureHemi, A)
if mult_safe
    @test_approx_eq F.L*F.L' A
end
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

A = rand(3,5); A = A'*A
F = cholfact(PureHemi, A)
if mult_safe
    @test_approx_eq F.L*F.L' A
end

# An indefinite matrix
A = [-1 0; 0 1]
F = cholfact(PureHemi, A)
@test_approx_eq F.L*F.L' A
b = rand(size(A,2))
x = F\b
@test_approx_eq x A\b

# A matrix that has no conventional LL' or LDL' factorization
A = [0 1; 1 0]
F = cholfact(PureHemi, A)
@test_approx_eq F.L*F.L' A
b = rand(size(A,2))
# low-level interface
indxsing = [1,2]
ytilde, Y = HemirealFactorizations.forwardsubst(F.L, b, indxsing)
@test ytilde == b*ν
@test Y == [μ 0; -ν μ]
α = rand(2)
yα1 = ytilde + Y*α
yα2 = HemirealFactorizations.forwardsubst(F.L, b, indxsing, α)
@test_approx_eq yα1 yα2
x, X, H, htildeα = HemirealFactorizations.backwardsubst(F.L, ytilde, Y, indxsing)
@test x == [0,0]
@test X == eye(2)
@test_approx_eq H -A
@test_approx_eq htildeα -b
α, Δb = HemirealFactorizations.resolve(H, htildeα, indxsing, b)
@test Δb == [0,0]  # because the matrix isn't singular
@test α == reverse(b)
# High-level interface
x = F\b
@test_approx_eq x A\b

# A matrix that would normally require pivoting
A = [0  1 -1;
     1  8 12;
    -1 12 20]
F = cholfact(PureHemi, A)
@test_approx_eq F.L*F.L' A
@test_approx_eq F.L [μ 0 0; ν 2μ+2ν 0; -ν 3μ+3ν μ+ν]
b = rand(size(A,2))
# low-level interface
ytilde, Y = HemirealFactorizations.forwardsubst(F.L, b)
@test_approx_eq ytilde [b[1]ν, b[2]/(2μ+2ν), (b[3]-3b[2]/2)/(μ+ν)]
@test_approx_eq Y [μ, -1/(2μ+2ν), 5/(2μ+2ν)]
xtilde, X, H, htildeα = HemirealFactorizations.backwardsubst(F.L, ytilde, Y)
# high-level interface
x = F\b
@test_approx_eq x A\b

# A singular matrix
A = rand(2,3); A = A'*A
b = rand(size(A,2))
xsvd = svdfact(A)\b
bsvd = A*xsvd
F = cholfact(PureHemi, A)
@test_approx_eq F.L*F.L' A
x = F\bsvd
@test_approx_eq_eps A*x bsvd 1e-10

# rhs not within the range of the matrix
x = F\b
bchol = A*x
Δb = bchol - b
# Test that the correction is limited to the 3rd entry
@test abs(Δb[1]) < 1e-12
@test abs(Δb[2]) < 1e-12
