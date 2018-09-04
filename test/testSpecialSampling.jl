# test special sampling function

using IncrementalInference, Distributions
import IncrementalInference: getSample

mutable struct ReuseData
  viewhdl::SubArray
  data::Vector{Complex{Float64}}
  ReuseData() = new(view(zeros(1),1,1), zeros(1)+im*0 )
end

struct EnhancedSamplingConditional <: FunctorPairwise
  mutabledata::Vector{ReuseData} # Threads.nthreads()
  staticdata::Array{Float64,2}
  specialSampling::Bool # trigger special enhanced getSample from IIF
  EnhancedSamplingConditional() = new()
  EnhancedSamplingConditional(sd::Array{Float64,2}) = new(ReuseData[ReuseData() for t in 1:Threads.nthreads()], sd, true)
  EnhancedSamplingConditional(md::Vector{ReuseData}, sd::Array{Float64,2}) = new(md, sd, true)
end

function getSample(esc::EnhancedSamplingConditional, userdata::FactorMetadata, N::Int=1)
  error("made it")
  return (randn(1,N), )
end


fg = emptyFactorGraph()

addNode!(fg, :x1, ContinuousScalar)
addFactor!(fg, [:x1], Prior(Normal()))

addNode!(fg, :x2, ContinuousScalar)

esc = EnhancedSamplingConditional(randn(5,10))
addFactor!(fg, [:x1], esc)





#
