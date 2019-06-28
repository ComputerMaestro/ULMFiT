"""
Weight-Dropped LSTM
"""

using Flux
import Flux: gate, tanh, σ, Tracker, params, gpu

# Generates Mask
dropMask(p, shape; type = Float32) = rand(type, shape...) .> p .* type(1/(1 - p))

# Converts vector of words to vector of one-hot vectors
onehot(wordVect::Vector, vocab::Vector) =
    oh_repr = broadcast(x -> (x ∈ vocab) ? Flux.onehot(x, vocab) : Flux.onehot("_unk_", vocab), wordVect)

#Adding "<pad>" keyowrd at the end if the length of the sentence is < bptt
function padding(batches::Vector)
    n = maximum([length(x) for x in batches])
    return ([length(batch) < n ? cat(batch, repeat(["<pos>"], n-length(batch)); dims = 1) : batch[1:n] for batch in batches], n)
end

# To initialize funciton for model LSTM weights
init_weights(extreme::AbstractFloat, dims...) = randn(Float32, dims...) .* sqrt(Float32(extreme))

#################### Weight-Dropped LSTM Cell#######################
mutable struct WeightDroppedLSTMCell{A, V}
    Wi::A
    Wh::A
    b::V
    h::V
    c::V
    p::Float64
    maskWi::BitArray
    maskWh::BitArray
end

function WeightDroppedLSTMCell(in::Integer, out::Integer, probability::Float64=0.0;
    init = Flux.glorot_uniform)
    cell = WeightDroppedLSTMCell(
        param(init(out*4, in)),
        param(init(out*4, out)),
        param(init(out*4)),
        param(zeros(Float32, out)),
        param(zeros(Float32, out)),
        probability,
        dropMask(probability, (out*4, in)),
        dropMask(probability, (out*4, out))
    )
    cell.b.data[gate(out, 2)] .= 1
    return cell
end

function (m::WeightDroppedLSTMCell)((h, c), x)
    b, o = m.b, size(h, 1)
    g = (m.Wi .* m.maskWi)*x .+ (m.Wh .* m.maskWh)*h .+ b
    input = σ.(gate(g, o, 1))
    forget = σ.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = σ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    return (h′, c), h′
end

Flux.@treelike WeightDroppedLSTMCell

# Weight-Dropped LSTM [stateful]
function WeightDroppedLSTM(a...; kw...)
    cell = WeightDroppedLSTMCell(a...;kw...)
    hidden = (cell.h, cell.c)
    return Flux.Recur(cell, hidden, hidden)
end

function resetMasks!(wd::T) where T <: Flux.Recur{<:WeightDroppedLSTMCell}
    wd.cell.maskWi = dropMask(wd.cell.p, size(wd.cell.Wi))
    wd.cell.maskWh = dropMask(wd.cell.p, size(wd.cell.Wi))
    return nothing
end
####################################################################

################## ASGD Weight-Dropped LSTM Layer###################
mutable struct AWD_LSTM
    layer::Flux.Recur
    T
    accum

    AWD_LSTM(a...; kw...) = new(WeightDroppedLSTM(a...; kw...))
end

(m::AWD_LSTM)(in) = m.layer(in)

setTrigger!(trigger_point, m::AWD_LSTM) = m.T = trigger_point;

params(m::AWD_LSTM) = Flux.params(m.layer)

function gpu(m::AWD_LSTM)
    ps = params(m)
    for p in ps
        p = Flux.gpu(p)
    end
    m.layer.cell.maskWi = Flux.gpu(m.layer.cell.maskWi)
    m.layer.cell.maskWh = Flux.gpu(m.layer.cell.maskWh)
    return m
end

# Averaged Stochastic Gradient Descent Step
function asgd_step!(iter, layer::AWD_LSTM)
    p = params(layer)
    if iter >= layer.T
        avg_fact = 1/max(iter - layer.T + 1, 1)
        if avg_fact != 1
            layer.accum = layer.accum + Tracker.data.(p)
            iter = 1
            for ps in p
                ps.data .= avg_fact*copy(layer.accum[iter])
                iter += 1
            end
        else
            layer.accumu = deepcopy(Tracker.data.(p))   # Accumulator for ASGD
        end
    end
end
####################################################################

########################## Varitional DropOut ######################
struct VarDrop{A <: AbstractArray, F <: AbstractFloat}
    mask::A
    p::F
end

VarDrop(probaility=0.0, shape) = VarDrop(gpu(dropMask(probability, shape)), probability)

(vd::VarDrop)(in) = in .* vd.mask

resetMasks!(vd::VarDrop) = (vd.mask = gpu(dropMask(vd.p, size(vd.mask))));
####################################################################

################# Varitional Dropped Embeddings ####################
mutable struct DroppedEmbeddings{A <: AbstractArray, T <: TrackedArray}
    emb::T
    p::Float64
    mask::A

    DroppedEmbeddings(in::Integer, embed_size::Integer, probability::Float64; init = Flux.glorot_uniform) =
        new(param(init(in, embed_size)), probability, dropMask(probability, (in, 1)))
end

(de::DroppedEmbeddings)(in::AbstractArray) = transpose((de.emb .* de.mask))*in

gpu(de::DroppedEmbeddings) = (de.emb = gpu(de.emb));

resetMasks!(de::DroppedEmbeddings) = (de.mask = dropMask(de.p, (size(de.emb, 1), 1)));

# Weight-tying
tiedEmbeddings(in::AbstractArray, de::DroppedEmbeddings) = (de.emb .* de.mask)*in
####################################################################
