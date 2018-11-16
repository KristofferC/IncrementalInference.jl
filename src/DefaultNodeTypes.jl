# default node types in IIF

SamplableBelief = Union{Distributions.Distribution, KernelDensityEstimate.BallTreeDensity, AliasingScalarSampler}

"""
Generic packed structure for the JSON factor serializers.
"""
mutable struct Packed_Factor
    measurement::Vector{Dict{String, Any}}
    distType::String
end

"""
One-dimensional scalar variable.
"""
struct ContinuousScalar <: InferenceVariable
  dims::Int
  labels::Vector{String}
  ContinuousScalar() = new(1, String[])
end
"""
One-dimensional multivariate variable.
"""
struct ContinuousMultivariate <: InferenceVariable
  dims::Int
  labels::Vector{String}
  ContinuousMultivariate() = new()
  ContinuousMultivariate(x) = new(x, String[])
end

"""
Unary factor representing a prior absolute measurement.
"""
struct Prior{T} <: IncrementalInference.FunctorSingleton where T <: SamplableBelief
  Z::T
end
# Testing constructor
Prior(::Type{FactorTestingFlag}) = Prior( MvNormal(zeros(3), 0.1*Matrix{Float64}(LinearAlgebra.I, 3,3)))
getSample(s::Prior, N::Int=1) = (reshape(rand(s.Z,N),:,N), )
"""
Converter: Prior -> Dict{String, Any}
"""
function convert(::Type{Dict{String, Any}}, prior::IncrementalInference.Prior)
    @warn "HERE!"
    z = convert(Dict{String, Any}, prior.Z)
    return JSON.parse(JSON.json(Packed_Factor([z], "Prior")))
end
"""
Converter: Dict{String, Any} -> Prior
"""
function convert(::Type{Prior}, prior::Dict{String, Any})
    # Genericize to any packed type next.
    z = prior["measurement"][1]
    z = convert(_evalType(z["distType"]), z)
    return Prior(z)
end

struct LinearConditional{T} <: IncrementalInference.FunctorPairwise where T <: SamplableBelief
  Z::T
end
getSample(s::LinearConditional, N::Int=1) = (reshape(rand(s.Z,N),:,N), )
function (s::LinearConditional)(res::Array{Float64},
                                userdata::FactorMetadata,
                                idx::Int,
                                meas::Tuple{Array{Float64, 2}},
                                X1::Array{Float64,2},
                                X2::Array{Float64,2}  )
  #
  res[1] = meas[1][idx] - (X2[1,idx] - X1[1,idx])
  nothing
end


struct MixturePrior{T} <: IncrementalInference.FunctorSingleton where {T <: SamplableBelief}
  Z::Vector{T}
  C::Distributions.Categorical
  MixturePrior{T}() where T  = new{T}()
  MixturePrior{T}(z::Vector{T}, c::Distributions.Categorical) where {T <: SamplableBelief} = new{T}(z, c)
  MixturePrior{T}(z::Vector{T}, p::Vector{Float64}) where {T <: SamplableBelief} = MixturePrior{T}(z, Distributions.Categorical(p))
end
MixturePrior(z::Vector{T}, c::Union{Distributions.Categorical, Vector{Float64}}) where {T <: SamplableBelief} = MixturePrior{T}(z, c)

getSample(s::MixturePrior, N::Int=1) = (reshape.(rand.(s.Z, N),1,:)..., rand(s.C, N))


struct MixtureLinearConditional{T} <: IncrementalInference.FunctorPairwise
  Z::Vector{T}
  C::Distributions.Categorical
  MixtureLinearConditional{T}() where T  = new{T}()
  MixtureLinearConditional{T}(z::Vector{T}, c::Distributions.Categorical) where {T <: SamplableBelief} = new{T}(z, c)
  MixtureLinearConditional{T}(z::Vector{T}, p::Vector{Float64}) where {T <: SamplableBelief} = MixtureLinearConditional{T}(z, Distributions.Categorical(p))
end
MixtureLinearConditional(z::Vector{T}, c::Union{Distributions.Categorical, Vector{Float64}}) where {T <: SamplableBelief} = MixtureLinearConditional{T}(z, c)

getSample(s::MixtureLinearConditional, N::Int=1) = (reshape.(rand.(s.Z, N),1,:)..., rand(s.C, N))
function (s::MixtureLinearConditional)(res::Array{Float64},
                               userdata::FactorMetadata,
                               idx::Int,
                               meas::Tuple,
                               X1::Array{Float64,2},
                               X2::Array{Float64,2}  )
  #
  res[1] = meas[meas[end][idx]][idx] - (X2[1,idx] - X1[1,idx])
  nothing
end



## packed types are still developed by hand.  Future versions would likely use a @packable macro to write Protobuf safe versions of factors

struct PackedPrior <: PackedInferenceType
  Z::String
  PackedPrior() = new()
  PackedPrior(z::AS) where {AS <: AbstractString} = new(z)
end
function convert(::Type{PackedPrior}, d::Prior)
  PackedPrior(string(d.Z))
end
function convert(::Type{Prior}, d::PackedPrior)
  Prior(extractdistribution(d.Z))
end


struct PackedLinearConditional <: PackedInferenceType
  Z::String
  PackedLinearConditional() = new()
  PackedLinearConditional(z::AS) where {AS <: AbstractString} = new(z)
end
function convert(::Type{PackedLinearConditional}, d::LinearConditional)
  PackedLinearConditional(string(d.Z))
end
function convert(::Type{LinearConditional}, d::PackedLinearConditional)
  LinearConditional(extractdistribution(d.Z))
end


struct PackedMixtureLinearConditional <: PackedInferenceType
  strs::Vector{String}
  cat::String
  PackedMixtureLinearConditional() = new()
  PackedMixtureLinearConditional(z::Vector{<:AbstractString}, cstr::AS) where {AS <: AbstractString} = new(z, cstr)
end
function convert(::Type{PackedMixtureLinearConditional}, d::MixtureLinearConditional)
  PackedMixtureLinearConditional(string.(d.Z), string(d.C))
end
function convert(::Type{MixtureLinearConditional}, d::PackedMixtureLinearConditional)
  MixtureLinearConditional(extractdistribution.(d.strs), extractdistribution(d.cat))
end



struct PackedMixturePrior <: PackedInferenceType
  strs::Vector{String}
  cat::String
  PackedMixturePrior() = new()
  PackedMixturePrior(z::Vector{<:AbstractString}, cstr::AS) where {AS <: AbstractString} = new(z, cstr)
end
function convert(::Type{PackedMixturePrior}, d::MixturePrior)
  PackedMixturePrior(string.(d.Z), string(d.C))
end
function convert(::Type{MixturePrior}, d::PackedMixturePrior)
  MixturePrior(extractdistribution.(d.strs), extractdistribution(d.cat))
end
