
# TODO stop-gap string storage of Distrubtion types, should be upgraded to more efficient storage
function normalfromstring(str::AS) where {AS <: AbstractString}
  meanstr = match(r"μ=[+-]?([0-9]*[.])?[0-9]+", str).match
  mean = split(meanstr, '=')[2]
  sigmastr = match(r"σ=[+-]?([0-9]*[.])?[0-9]+", str).match
  sigma = split(sigmastr, '=')[2]
  Normal{Float64}(parse(Float64,mean), parse(Float64,sigma))
end

function mvnormalfromstring(str::AS) where {AS <: AbstractString}
  means = split(split(split(str, 'μ')[2],']')[1],'[')[end]
  mean = Float64[]
  for ms in split(means, ',')
    push!(mean, parse(Float64, ms))
  end
  sigs = split(split(split(str, 'Σ')[2],']')[1],'[')[end]
  sig = Float64[]
  for ms in split(sigs, ';')
    for m in split(ms, ' ')
      length(m) > 0 ? push!(sig, parse(Float64, m)) : nothing
    end
  end
  len = length(mean)
  sigm = reshape(sig, len,len)
  MvNormal(mean, sigm)
end

function categoricalfromstring(str::AS)::Distributions.Categorical where {AS <: AbstractString}
  # pstr = match(r"p=\[", str).match
  psubs = split(str, '=')[end]
  psubs = split(psubs, '[')[end]
  psubsub = split(psubs, ']')[1]
  pw = split(psubsub, ',')
  p = parse.(Float64, pw)
  return Categorical(p ./ sum(p))
end

function extractdistribution(str::AS)::Union{Nothing, SamplableBelief} where {AS <: AbstractString}
  # TODO improve use of multidispatch and packing of Distribution types
  if str == ""
    return nothing
  elseif (occursin(r"Normal", str) && !occursin(r"FullNormal", str))
    return normalfromstring(str)
  elseif occursin(r"FullNormal", str)
    return mvnormalfromstring(str)
  elseif occursin(r"Categorical", str)
    return categoricalfromstring(str)
  elseif occursin(r"KDE:", str)
    return convert(BallTreeDensity, str)
  elseif occursin(r"AliasingScalarSampler", str)
    return convert(AliasingScalarSampler, str)
  else
    error("Don't know how to extract distribution from str=$(str)")
  end
end

function extractdistributionJson(packedFactor::Dict{String, Any})::Union{Void, Distributions.Distribution}
  # TODO improve use of multidispatch and packing of Distribution types
  error("Not implemented yet")
end

## JSON distribution serialization ##

"""
Converter: Packed_MvNormal -> MvNormal
"""
function convert(::Type{Distributions.MvNormal}, pv::Dict{String, Any})
    len = length(Float64.(pv["mean"]))
    mat = reshape(Float64.(pv["cov"]), len, len)
    return Distributions.MvNormal(Float64.(pv["mean"]), mat)
end

"""
Converter: MvNormal -> Packed_MvNormal
"""
function convert(::Type{Dict{String, Any}}, mvNormal::Distributions.MvNormal)
    v = mvNormal.Σ.mat[:]
    return JSON.parse(JSON.json(Packed_MvNormal(mvNormal.μ, v, "MvNormal")))
end

"""
Converter: Packed_Normal -> Normal
"""
function convert(::Type{Distributions.Normal}, pv::Dict{String, Any})
    return Distributions.Normal(Float64(pv["mean"]), Float64(pv["std"]))
end

"""
Converter: Normal -> Packed_Normal
"""
function convert(::Type{Dict{String, Any}}, normal::Distributions.Normal)
    return JSON.parse(JSON.json(Packed_Normal(normal.μ, normal.σ, "Normal")))
end


"""
Converter: Packed_AliasingScalarSampler -> AliasingScalarSampler
"""
function convert(::Type{IncrementalInference.AliasingScalarSampler}, pv::Dict{String, Any})
    sampler = IncrementalInference.AliasingScalarSampler(Float64.(pv["samples"]), Float64.(pv["weights"]); SNRfloor=pv["quantile"])
    return sampler
end

"""
Converter: AliasingScalarSampler -> Packed_AliasingScalarSampler
"""
function convert(::Type{Packed_AliasingScalarSampler}, sampler::IncrementalInference.AliasingScalarSampler)
    packed = Packed_AliasingScalarSampler(sampler.domain, sampler.weights.values, 0.0, "AliasingScalarSampler")
    return JSON.parse(JSON.json(packed))
end
