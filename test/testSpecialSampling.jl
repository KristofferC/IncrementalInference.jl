# test special sampling function

using IncrementalInference, Distributions
import IncrementalInference: getSample

mutable struct ReuseData
  viewhdl::SubArray
  data::Vector{Complex{Float64}}
  ReuseData() = new(view(zeros(1),1,1), zeros(1)+im*0 )
end

struct EnhancedSamplingConditional <: FunctorPairwise
  mutabledata::Vector{ReuseData} # one for each Threads.nthreads()
  staticdata::Array{Float64,2}
  specialSampling::Bool # trigger special enhanced getSample from IIF
  EnhancedSamplingConditional() = new()
  EnhancedSamplingConditional(sd::Array{Float64,2}) = new(ReuseData[ReuseData() for t in 1:Threads.nthreads()], sd, true)
  EnhancedSamplingConditional(md::Vector{ReuseData}, sd::Array{Float64,2}) = new(md, sd, true)
end

function getSample!(meas::Tuple{Array{Float64,2}}, esc::EnhancedSamplingConditional, userdata::FactorMetadata, N::Int=1)
  return (randn(1,N), )
  nothing
end
function getSample(esc::EnhancedSamplingConditional, N::Int=1)
  error("SHOULD NOT HAPPEN!!!")
end

function (es::EnhancedSamplingConditional)(res::Vector{Float64},
                                           userdata::FactorMetadata,
                                           idx::Int,
                                           meas::Tuple{Array{Float64,2}},
                                           Xi::A,
                                           Xj::A  ) where {A <: AbstractArray}
  #
  res[1] = meas[1][idx] - (Xj[1,idx] - Xi[1,idx])
  nothing
end

fg = emptyFactorGraph()

addNode!(fg, :x1, ContinuousScalar)
addFactor!(fg, [:x1], Prior(Normal()))

addNode!(fg, :x2, ContinuousScalar)

esc = EnhancedSamplingConditional(randn(5,10))
addFactor!(fg, [:x1], esc)





#
