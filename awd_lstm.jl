"""
Weight-Dropped LSTM
"""

using Flux
import Flux: gate, tanh, σ, Tracker, params, gpu

# Weight-Dropped LSTM Cell
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
        rand(Float32, out*4, in) .> probability,
        rand(Float32, out*4, in) .> probability,
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
    wd.cell.maskWi = gpu(rand(Float32, size(wd.cell.Wi)...)) .> wd.cell.p
    wd.cell.maskWh = gpu(rand(Float32, size(wd.cell.Wh)...)) .> wd.cell.p
    return nothing
end

# ASGD Weight-Dropped LSTM Layer
mutable struct AWD_LSTM
    T
    layer::Flux.Recur
    accum

    AWD_LSTM(trigger_iter, a...; kw...) = new(
                                            trigger_iter,
                                            WeightDroppedLSTM(a...; kw...)
                                        )
end

(m::AWD_LSTM)(in) = m.layer(in)

params(m::AWD_LSTM) = Flux.params(m.layer)
function gpu(m::AWD_LSTM)
    ps = params(m)
    for p in ps
        p = Flux.gpu(p)
    end
    m.maskWi = Flux.gpu(m.mask)
    m.maskWh = Flux.gpu()
end

# Averaged Stochastic Gradient Descent Step
function asgd_step(iter, layer::AWD_LSTM)
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
