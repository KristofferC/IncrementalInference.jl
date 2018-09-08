# development testing for multithreaded convolution

using Distributions
using IncrementalInference
using Base: Test

import IncrementalInference: getSample



@testset "Basic ContinuousScalar example to ensure multithreaded convolutions work..." begin

@show Threads.nthreads()


N = 100

# Start with an empty factor graph
fg = emptyFactorGraph()

# add the first node
addNode!(fg, :x0, ContinuousScalar, N=N)

# this is unary (prior) factor and does not immediately trigger autoinit of :x0.
addFactor!(fg, [:x0], Prior(Normal(0,1)))


addNode!(fg, :x1, ContinuousScalar, N=N)
# P(Z | :x1 - :x0 ) where Z ~ Normal(10,1)
addFactor!(fg, [:x0, :x1], LinearConditional(Normal(10.0,1)), threadmodel=SingleThreaded)


pts = approxConv(fg, :x0x1f1, :x1, N=N)

@test 0.95*N <= sum(abs.(pts - 10.0) .< 5.0)

##


@test isInitialized(fg, :x0)
# isInitialized(fg, :x1)
doautoinit!(fg, :x1)

fnc = getData(fg, :x0x1f1, nt=:fnc)
ccw = fnc.fnc

@test ccw.xDim == 1
@test ccw.zDim == 1
@test size(ccw.cpt[1].X) == (1,N)
@test length(ccw.cpt[1].Y) == 1
@test length(ccw.cpt[1].perturb) == 1
@test length(ccw.params) == 2
@test norm(ccw.params[1] - getVal(fg, :x0)) < 1e-4
@test norm(ccw.params[2] - getVal(fg, :x1)) < 1e-4

@test ccw.hypotheses == nothing
@test sum(ccw.certainhypo-[1:2;]) < 1e-5

res = zeros(1)

ccw.cpt[1].particleidx = 1
ccw.cpt[1].activehypo = [1:2;]

ccw(res)

@show abs(res[1]) < 1e-5

# fieldnames(ccw)
# getData(fc).fnc.xDim
# getData(fc).fnc.measurement


end


#
