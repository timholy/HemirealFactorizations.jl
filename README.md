# HemiplexFactorizations

[![CI](https://github.com/timholy/HemiplexFactorizations.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/timholy/HemiplexFactorizations.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/timholy/HemiplexFactorizations.jl/graph/badge.svg?token=ldfY1pgacu)](https://codecov.io/gh/timholy/HemiplexFactorizations.jl)

# Introduction

Cholesky factorizations over the hemiplex numbers can be computed for
arbitrary symmetric matrices, including indefinite and singular
matrices.  For singular matrices, the behavior is reminiscent of the
singular value decomposition, but the performance is much better.

# Usage

After creating a symmetric matrix `A`, compute its Cholesky
factorization over the hemiplex numbers like this:

```jl
F = cholfact(PureHemi, A)
```
Then you can use `F` to solve equations, e.g.,
```jl
x = F \ b
```
If `A` has zero pivots, you will need to use `x = nullsolver(F) \ b` instead.

If `A` is singular, this should be the least-squares solution.

## Supported operations

You can compute `F*F'` or say `rank(F)`.  You can also convert `F`
into matrix form with `convert(Matrix, F)`.

## Options

```jl
F = cholfact(PureHemi, A, δ; blocksize=default)
```
where:

- `δ` is the tolerance on the diagonal values of `A` during factorization; any with magnitudes smaller than `δ` will be treated as if they are 0.
- `blocksize` controls the performance of the factorization algorithm.
