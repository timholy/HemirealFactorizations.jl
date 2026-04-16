module HemirealFactorizations

using LinearAlgebra
using HemirealNumbers
using SparseArrays

export nullsolver
export HemiCholesky, HemiCholeskyReal, HemiCholeskyPivot, HemiCholeskyXY, SparseHemiCholeskyReal

include("cholesky.jl")
include("sparse_cholesky.jl")

(*)(L::LowerTriangular{PureHemi{T}, <:AbstractMatrix{PureHemi{T}}}, y::AbstractVector{PureHemi{T}}) where T = mul!(similar(y, T), L, y)
(*)(U::UpperTriangular{PureHemi{T}, <:Adjoint{PureHemi{T}, <:AbstractMatrix{PureHemi{T}}}}, x::AbstractVector{<:Real}) where T = mul!(similar(x, PureHemi{T}), U, x)
*(A::LowerTriangular{PureHemi{S}}, B::UpperTriangular{PureHemi{T}}) where {S<:Real,T<:Real} = mul!(similar(B, promote_type(S,T), size(B)), A, B)

(\)(L::LowerTriangular{PureHemi{T}, <:AbstractMatrix{PureHemi{T}}}, b::AbstractVector) where T = forwardsubst!(similar(b, PureHemi{T}), L, b)
(\)(U::UpperTriangular{PureHemi{T}, <:Adjoint{PureHemi{T}, <:AbstractMatrix{PureHemi{T}}}}, y::AbstractVector{<:PureHemi}) where T = backwardsubst!(similar(y, T), Matrix{T}(undef, 0, 0), U', y)

end # module
