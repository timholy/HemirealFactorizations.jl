# HemirealFactorizations

[![CI](https://github.com/timholy/HemirealFactorizations.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/timholy/HemirealFactorizations.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/timholy/HemirealFactorizations.jl/graph/badge.svg?token=ldfY1pgacu)](https://codecov.io/gh/timholy/HemirealFactorizations.jl)

# Introduction

Cholesky factorizations over the hemireals can be computed for
arbitrary symmetric matrices, including singular and indefinite
matrices.  For singular matrices, the behavior is reminiscent of the
singular value decomposition, but the performance is much better.

# Usage

After creating a symmetric matrix `A`, compute its Cholesky
factorization over the hemireal numbers like this:

```jl
F = cholfact(PureHemi, A)
```
Then you can use `F` to solve equations, e.g.,
```jl
x = F\b
```
If `A` is singular, this should be the least-squares solution.

## Supported operations

You can compute `F*F'` or say `rank(F)`.  You can also convert `F`
into matrix form with `convert(Matrix, F)`.

To support all operations, you need to be running at least a version
of julia-0.5-dev that is current with master as of 2015-12-11.
However, many operations also work on julia-0.4.

## Options

```jl
F = cholfact(PureHemi, A, δ; blocksize=default)
```
where:

- `δ` is the tolerance on the diagonal values of `A` during factorization; any with magnitudes smaller than `δ` will be treated as if they are 0.
- `blocksize` controls the performance of the factorization algorithm.
