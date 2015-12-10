# HemirealFactorizations

[![Build Status](https://travis-ci.org/timholy/HemirealFactorizations.jl.svg?branch=master)](https://travis-ci.org/timholy/HemirealFactorizations.jl)
[![codecov.io](https://codecov.io/github/timholy/HemirealFactorizations.jl/coverage.svg?branch=master)](https://codecov.io/github/timholy/HemirealFactorizations.jl?branch=master)

## Optimizations

The implementation here is intended to be exploratory/instructional rather than of maximal efficiency.  For a "workhorse" algorithm, some of the changes would be:

- Replace operations using pure-hemireals (which involve two components) by real numbers (single components), exploiting analytic results to obtain the other component.  Since we only have a single component, this necessitates deleting all sanity-check assertions.
- In-place storage: with a little bit of bookkeeping, one could store `L` in `A` (as long as `A` isn't, e.g., `Matrix{Int}`).
