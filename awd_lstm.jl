"""
ASGD Weight-Dropped LSTM
"""

using Flux
import Flux: gate, tanh, σ, Tracker, params, gpu, cpu

# Generates Mask
dropMask(p, shape; type = Float32) = (rand(type, shape...) .> p) .* type(1/(1 - p))

#################### Weight-Dropped LSTM Cell#######################
mutable struct WeightDroppedLSTMCell{A, V, M}
    Wi::A
    Wh::A
    b::V
    h::V
    c::V
    p::Float64
    maskWi::M
    maskWh::M
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

function reset_masks!(wd::T) where T <: Flux.Recur{<:WeightDroppedLSTMCell}
    wd.cell.maskWi = (typeof(wd.cell.maskWi) <: Array) ? dropMask(wd.cell.p, size(wd.cell.Wi)) : gpu(dropMask(wd.cell.p, size(wd.cell.Wi)))
    wd.cell.maskWh = (typeof(wd.cell.maskWh) <: Array) ? dropMask(wd.cell.p, size(wd.cell.Wh)) : gpu(dropMask(wd.cell.p, size(wd.cell.Wh)))
    return nothing
end
####################################################################

################## ASGD Weight-Dropped LSTM Layer###################
mutable struct AWD_LSTM
    layer::Flux.Recur
    T::Integer
    accum
end

AWD_LSTM(in::Integer, out::Integer, probability::Float64=0.0; kw...) = AWD_LSTM(WeightDroppedLSTM(in, out, probability; kw...), -1, [])

Flux.@treelike AWD_LSTM

(m::AWD_LSTM)(in) = m.layer(in)

setTrigger!(trigger_point, m::AWD_LSTM) = m.T = trigger_point;

function gpu!(m::AWD_LSTM)
    m.layer = gpu(m.layer)
    return
end

function cpu!(m::AWD_LSTM)
    m.layer = cpu(m.layer)
    return
end

reset_masks!(awd::AWD_LSTM) = reset_masks!(awd.layer)

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

"""
Variational Dropout
"""

########################## Varitional DropOut ######################
mutable struct VarDrop{F <: AbstractFloat, A <: AbstractArray}
    p::F
    mask::A
end

VarDrop(shape, probability=0.0) = VarDrop{Float64, AbstractArray}(probability, dropMask(probability, shape))

(vd::VarDrop)(in) = in .* vd.mask

function reset_masks!(vd::VarDrop)
    vd.mask = (typeof(vd.mask) <: Array) ? dropMask(vd.p, size(vd.mask)) : gpu(dropMask(vd.p, size(vd.mask)))
    return nothing
end

function gpu!(vd::VarDrop)
    vd.mask = gpu(vd.mask);
    return
end

function cpu!(vd::VarDrop)
    vd.mask = cpu(vd.mask);
    return
end
####################################################################

"""
Embeddings with varitional dropout
"""

################# Varitional Dropped Embeddings ####################
mutable struct DroppedEmbeddings{A}
    emb::TrackedArray
    p::Float64
    mask::A
end

DroppedEmbeddings(in::Integer, embed_size::Integer, probability::Float64=0.0;
    init = Flux.glorot_uniform) =
        DroppedEmbeddings{AbstractArray}(
            param(init(in, embed_size)),
            probability,
            dropMask(probability, (in, 1))
        )

function (de::DroppedEmbeddings)(in::AbstractArray, tying=false)
    dropped = de.emb .* de.mask
    return tying ? dropped * in : transpose(dropped[in, :])
end

Flux.@treelike DroppedEmbeddings

function gpu!(de::DroppedEmbeddings)
    de.emb = gpu(de.emb)
    de.mask = gpu(de.mask)
    return
end

function cpu!(de::DroppedEmbeddings)
    de.emb = cpu(de.emb)
    de.mask = cpu(de.mask)
    return
end

function reset_masks!(de::DroppedEmbeddings)
    de.mask = (typeof(de.mask) <: Array) ? dropMask(de.p, size(de.mask)) : gpu(dropMask(de.p, size(de.mask)))
    return nothing
end
####################################################################

# Reset's dropping probability
function reset_probability!(new_p, m::WeightDroppedLSTMCell)
    m.p = new_p
    return nothing
end

reset_probability!(new_p, m::Flux.Recur{<:WeightDroppedLSTMCell}) = reset_probability!(new_p, m.cell)
reset_probability!(new_p, m::AWD_LSTM) = reset_probability!(new_p, m.layer.cell)

function reset_probability!(new_p, m::VarDrop)
    m.p = new_p
    return nothing
end

function reset_probability!(new_p, m::DroppedEmbeddings)
    m.p = new_p
    return nothing
end
