


const BTGdict = GenericIncidenceList{ExVertex,Edge{ExVertex},Array{ExVertex,1},Array{Array{Edge{ExVertex},1},1}}

# BayesTree declarations
"""
$(TYPEDEF)

Data structure for the Bayes (Junction) tree, which is used for inference and constructed from a given `::FactorGraph`.
"""
mutable struct BayesTree
  bt::BTGdict
  btid::Int
  cliques::Dict{Int,TreeClique}
  frontals::Dict{Symbol,Int}
  variableOrder::Vector{Symbol}
  buildTime::Float64
end

function emptyBayesTree()
    bt =   BayesTree(Graphs.inclist(TreeClique,is_directed=true),
                     0,
                     Dict{Int,TreeClique}(),
                     #[],
                     Dict{AbstractString, Int}(),
                     Symbol[],
					 0.0 )
    return bt
end


"""
    $TYPEDEF

Container for upward tree solve / initialization.

TODO
- remove proceed
- more direct clique access (cliq, parent, children), for multi-process solves
"""
mutable struct CliqStateMachineContainer{BTND, T <: AbstractDFG, InMemG <: InMemoryDFGTypes}
  dfg::T
  cliqSubFg::InMemG
  tree::BayesTree
  cliq::TreeClique
  parentCliq::Vector{TreeClique}
  childCliqs::Vector{TreeClique}
  forceproceed::Bool # TODO: bad flag that must be removed by refactoring sm
  incremental::Bool
  drawtree::Bool
  dodownsolve::Bool
  delay::Bool
  opts::SolverParams
  refactoring::Dict{Symbol, String}
  oldcliqdata::BTND
  logger::SimpleLogger
  CliqStateMachineContainer{BTND}() where {BTND} = new{BTND, InMemDFGType, InMemDFGType}() # NOTE JT - GraphsDFG as default?
  CliqStateMachineContainer{BTND}(x1::G,
                                  x2::InMemoryDFGTypes,
                                  x3::BayesTree,
                                  x4::TreeClique,
                                  x5::Vector{TreeClique},
                                  x6::Vector{TreeClique},
                                  x7::Bool,
                                  x8::Bool,
                                  x9::Bool,
                                  x10a::Bool,
                                  x10aa::Bool,
                                  x10aaa::SolverParams,
								  x10b::Dict{Symbol,String}=Dict{Symbol,String}(),
                                  x11::BTND=emptyBTNodeData(),
                                  x13::SimpleLogger=SimpleLogger(Base.stdout) ) where {BTND, G <: AbstractDFG} = new{BTND, G, typeof(x2)}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10a,x10aa,x10aaa,x10b,x11, x13)
end

function CliqStateMachineContainer(x1::G,
                                   x2::InMemoryDFGTypes,
                                   x3::BayesTree,
                                   x4::TreeClique,
                                   x5::Vector{TreeClique},
                                   x6::Vector{TreeClique},
                                   x7::Bool,
                                   x8::Bool,
                                   x9::Bool,
                                   x10::Bool,
                                   x10aa::Bool,
                                   x10aaa::SolverParams,
                                   x11::BTND=emptyBTNodeData(),
                                   x13::SimpleLogger=SimpleLogger(Base.stdout) ) where {BTND, G <: AbstractDFG}
  #
  CliqStateMachineContainer{BTND}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x10aa,x10aaa,Dict{Symbol,String}(),x11,x13)
end

const CSMHistory = Vector{Tuple{DateTime, Int, Function, CliqStateMachineContainer}}



"""
$(TYPEDEF)

Data structure for each clique in the Bayes (Junction) tree.
"""
mutable struct BayesTreeNodeData
  frontalIDs::Vector{Symbol}
  separatorIDs::Vector{Symbol}
  inmsgIDs::Vector{Symbol} # Int
  potIDs::Vector{Symbol} # Int # this is likely redundant TODO -- remove
  potentials::Vector{Symbol}
  partialpotential::Vector{Bool}

  dwnPotentials::Vector{Symbol}
  dwnPartialPotential::Vector{Bool}

  cliqAssocMat::Array{Bool,2}
  cliqMsgMat::Array{Bool,2}
  directvarIDs::Vector{Symbol} # Int
  directFrtlMsgIDs::Vector{Symbol} # Int
  msgskipIDs::Vector{Symbol} # Int
  itervarIDs::Vector{Symbol} # Int
  directPriorMsgIDs::Vector{Symbol} # Int
  debug
  debugDwn

  # future might concentrate these four fields down to two
  # these should become specialized BeliefMessage type
  upMsg::TempBeliefMsg # Dict{Symbol, BallTreeDensity}
  dwnMsg::TempBeliefMsg # Dict{Symbol, BallTreeDensity}
  upInitMsgs::Dict{Int, TempBeliefMsg}
  downInitMsg::TempBeliefMsg

  allmarginalized::Bool
  initialized::Symbol
  upsolved::Bool
  downsolved::Bool
  initUpChannel::Channel{Symbol}
  initDownChannel::Channel{Symbol}
  solveCondition::Condition
  lockUpStatus::Channel{Int}
  lockDwnStatus::Channel{Int}
  solvableDims::Channel{Dict{Symbol, Float64}}
  statehistory::Vector{Tuple{DateTime, Int, Function, CliqStateMachineContainer}}
  BayesTreeNodeData() = new()
  BayesTreeNodeData(x...) = new(x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],
                                x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],
                                x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31],
                                Vector{Tuple{DateTime, Int, Function, CliqStateMachineContainer}}() )
end

# TODO -- this should be a constructor
function emptyBTNodeData()
  BayesTreeNodeData(Symbol[],Symbol[],Symbol[],
                    Symbol[],Symbol[],Bool[], # 6
                    Symbol[],Bool[],
                    Array{Bool}(undef, 0,0),
                    Array{Bool}(undef, 0,0),
                    Int[],Int[],             # 10+2
                    Int[],Int[],Int[],       # 13+2
                    nothing, nothing,        # 15+2
                    Dict{Symbol, BallTreeDensity}(),  # :null => AMP.manikde!(zeros(1,1), [1.0;], (:Euclid,))),
                    Dict{Symbol, BallTreeDensity}(),  # :null => AMP.manikde!(zeros(1,1), [1.0;], (:Euclid,))),
                    Dict{Int, TempBeliefMsg}(),
                    TempBeliefMsg(),         # 19+2
                    false, :null,
                    false, false,            # 23+2
                    Channel{Symbol}(1), Channel{Symbol}(1), Condition(), # 26+2
                    Channel{Int}(1), Channel{Int}(1),
                    Channel{Dict{Symbol,Float64}}(1) )
end



"""
$(TYPEDEF)
"""
mutable struct PotProd
    Xi::Symbol # Int
    prev::Array{Float64,2}
    product::Array{Float64,2}
    potentials::Array{BallTreeDensity,1}
    potentialfac::Vector{Symbol}
end
"""
$(TYPEDEF)
"""
mutable struct CliqGibbsMC
    prods::Array{PotProd,1}
    lbls::Vector{Symbol}
    CliqGibbsMC() = new()
    CliqGibbsMC(a,b) = new(a,b)
end
"""
$(TYPEDEF)
"""
mutable struct DebugCliqMCMC
  mcmc::Union{Nothing, Array{CliqGibbsMC,1}}
  outmsg::NBPMessage
  outmsglbls::Dict{Symbol, Symbol} # Int
  priorprods::Vector{CliqGibbsMC}
  DebugCliqMCMC() = new()
  DebugCliqMCMC(a,b,c,d) = new(a,b,c,d)
end

"""
$(TYPEDEF)
"""
mutable struct UpReturnBPType
  upMsgs::NBPMessage
  dbgUp::DebugCliqMCMC
  IDvals::Dict{Symbol, EasyMessage}
  keepupmsgs::TempBeliefMsg # Dict{Symbol, BallTreeDensity} # TODO Why separate upMsgs?
  totalsolve::Bool
  UpReturnBPType() = new()
  UpReturnBPType(x1,x2,x3,x4,x5) = new(x1,x2,x3,x4,x5)
end

"""
$(TYPEDEF)

TODO refactor msgs into only a single variable
"""
mutable struct DownReturnBPType
  dwnMsg::NBPMessage
  dbgDwn::DebugCliqMCMC
  IDvals::Dict{Symbol,EasyMessage} # Int
  keepdwnmsgs::TempBeliefMsg # Dict{Symbol, BallTreeDensity}
end

"""
$(TYPEDEF)
"""
mutable struct FullExploreTreeType{T, T2, T3 <:InMemoryDFGTypes}
  fg::T3
  bt::T2
  cliq::TreeClique
  prnt::T
  sendmsgs::Vector{NBPMessage}
end

const ExploreTreeType{T} = FullExploreTreeType{T, BayesTree}
const ExploreTreeTypeLight{T} = FullExploreTreeType{T, Nothing}


function ExploreTreeType(fgl::G,
                         btl::BayesTree,
                         vertl::TreeClique,
                         prt::T,
                         msgs::Array{NBPMessage,1} ) where {G <: AbstractDFG, T}
  #
  ExploreTreeType{T}(fgl, btl, vertl, prt, msgs)
end

"""
$(TYPEDEF)
"""
mutable struct MsgPassType
  fg::GraphsDFG
  cliq::TreeClique
  vid::Symbol # Int
  msgs::Array{NBPMessage,1}
  N::Int
end




mutable struct PackedBayesTreeNodeData
  frontalIDs::Vector{Symbol}
  separatorIDs::Vector{Symbol}
  inmsgIDs::Vector{Symbol} # Int
  potIDs::Vector{Symbol} # Int # this is likely redundant TODO -- remove
  potentials::Vector{Symbol}
  partialpotential::Vector{Bool}
  dwnPotentials::Vector{Symbol}
  dwnPartialPotential::Vector{Bool}
  cliqAssocMat::Array{Bool,2}
  cliqMsgMat::Array{Bool,2}
  directvarIDs::Vector{Symbol} # Int
  directFrtlMsgIDs::Vector{Symbol} # Int
  msgskipIDs::Vector{Symbol} # Int
  itervarIDs::Vector{Symbol} # Int
  directPriorMsgIDs::Vector{Symbol} # Int
end



function convert(::Type{PackedBayesTreeNodeData}, btnd::BayesTreeNodeData)
  return PackedBayesTreeNodeData(
    btnd.frontalIDs,
    btnd.separatorIDs,
    btnd.inmsgIDs,
    btnd.potIDs,
    btnd.potentials,
    btnd.partialpotential,
    btnd.dwnPotentials,
    btnd.dwnPartialPotential,
    btnd.cliqAssocMat,
    btnd.cliqMsgMat,
    btnd.directvarIDs,
    btnd.directFrtlMsgIDs,
    btnd.msgskipIDs,
    btnd.itervarIDs,
    btnd.directPriorMsgIDs  )
end


function convert(::Type{BayesTreeNodeData}, pbtnd::PackedBayesTreeNodeData)
  btnd = emptyBTNodeData()
    btnd.frontalIDs = pbtnd.frontalIDs
    btnd.separatorIDs = pbtnd.separatorIDs
    btnd.inmsgIDs = pbtnd.inmsgIDs
    btnd.potIDs = pbtnd.potIDs
    btnd.potentials = pbtnd.potentials
    btnd.partialpotential = pbtnd.partialpotential
    btnd.dwnPotentials = pbtnd.dwnPotentials
    btnd.dwnPartialPotential = pbtnd.dwnPartialPotential
    btnd.cliqAssocMat = pbtnd.cliqAssocMat
    btnd.cliqMsgMat = pbtnd.cliqMsgMat
    btnd.directvarIDs = pbtnd.directvarIDs
    btnd.directFrtlMsgIDs = pbtnd.directFrtlMsgIDs
    btnd.msgskipIDs = pbtnd.msgskipIDs
    btnd.itervarIDs = pbtnd.itervarIDs
    btnd.directPriorMsgIDs = pbtnd.directPriorMsgIDs
  return btnd
end
