"""
ULMFiT - FINE-TUNING
"""

using Flux
using Flux: Tracker
using CorpusLoaders

cd(@__DIR__)
include("pretrainLM.jl")    # importing LanguageModel and useful functions
include("awd_lstm.jl")      # importing AWD_LSTM, VarDrop and DroppedEmbeddings
include("utils.jl")         # importing utilities

function data_loader(filepath::String)
    targetData = read(open(filepath, "r"), String)
    return intern.(tokenize(targetData))
end

function discriminative_step!(layers, p::Flux.Params, ηL::Float64, l, gradient_clip::Float64)
    # Applying gradient clipping
    l = Tracker.hook(x -> grad_clipping(x, gradient_clip), l)

    # Gradient calculation
    grads = Tracker.gradient(() -> l, p)

    # discriminative step
    ηl = ηL/(2.6^length(layers))
    for layer in layers
        ηl *= 2.6
        for ps in params(layer)
            Tracker.update!(ps, -ηl*grad[ps])
        end
    end
    return nothing
end

# Fine-Tuning Language Model
function fineTuneLM(lm::LanguageModel; batchsize::Integer=64, bptt::Integer=70, gradient_clip::Float64=0.25,
        ηL::Float64=4e-3, stlr_cut_frac::Float64=0.1, stlr_η_max::Float64=0.01, stlr_ratio::Float32=32,  α::Number=2, β::Number=1, epochs::Integer=1, checkpoint_iter::Integer=5000)
    gen = Channel(x -> generator(x, data_loader(); batchsize=batchsize, bptt=bptt))
    num_of_iters = take!(gen)
    T = Int(floor((num_of_iters*2)/100))
    p = params(lm)

    model_layers = Chain(
        lm.embedding_layer,
        VarDrop((size(lm.embedding_layer.emb, 2), batchsize), lm.wordDropProb),
        lm.lstm_layer1,
        VarDrop((size(lm.lstm_layer1.layer.cell.h, 1), batchsize), lm.LayerDropProb),
        lm.lstm_layer2,
        VarDrop((size(lm.lstm_layer2.layer.cell.h, 1), batchsize), lm.LayerDropProb),
        lm.lstm_layer3,
        VarDrop((size(lm.lstm_layer3.layer.cell.h, 1), batchsize), lm.FinalDropProb),
        x -> lm.embedding_layer(x, true),
        softmax
    )

    # Fine-Tuning loops
    for epoch=1:epochs
        println("\nEpoch: $epoch")
        for i=1:num_of_iters

            # FORWARD
            l = loss(lm, model_layers, gen, α, β)

            # Slanted triangular learning rate step
            cut = T * stlr_cut_frac
            p_frac = (i < cut) ? i/cut : (1 - ((i-cut)/(cut*(1/stlr_cut_frac-1))))
            ηL = stlr_η_max*((1+p_frac*(stlr_ratio-1))/stlr_ratio)

            # Backprop with discriminative fine-tuning step
            discriminative_step!(model_layers[[1, 3, 5, 7]], p, ηL, l, gradient_clip)

            # ASGD Step, after Triggering
            asgd_step!.(i, [lm.lstm_layer1,lm.lstm_layer2,lm.lstm_layer3])

            println("loss: $l", " iteration number: $i")

            # Saving checkpoints
            if i == checkpoint_iter save_model!(lm) end
        end
    end
end
