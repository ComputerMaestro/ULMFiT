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

function discriminative_step!(layers, ηL::Float64, l, gradient_clip::Float64)
    # Gradient calculation
    grads = Tracker.gradient(() -> l, p)
    for ps in p     # Applying Gradient clipping
        grads[ps].data = min.(grads[ps].data, gradient_clip)
    end

    # discriminative step
    ηl = ηL/(2.6^length(layers))
    for layer in layers
        ηl *= 2.6
        for p in params(layer)
            Tracker.update!(p, -ηl*grad[p])
        end
    end
    return nothing
end

# Fine-Tuning Language Model
function fineTuneLM(lm::LanguageModel; batchsize::Integer=70, bptt::Integer=70, gradient_clip::Float64=0.25,
        ηL::Float64=4e-3, stlr_cut_frac::Float64=0.1, stlr_ratio::Float32=32,  α::Number=2, β::Number=1, epochs::Integer=1, checkpoint_iter::Integer=5000)
    gen = Channel(x -> generator(x, data_loader(); batchsize=batchsize, bptt=bptt))
    num_of_iters = take!(gen)
    T = Int(floor((num_of_iters*2)/100))
    p = params(lm)
    H_prev = repeat([zeros(Float32, length(lm.vocab), batchsize)], bptt)

    model_layers = Chain(
        lm.embedding_layer,
        VarDrop((size(lm.embedding_layer.emb, 2), batchsize), lm.wordDropProb),
        lm.lstm_layer1,
        VarDrop((size(lm.lstm_layer1.layer.cell.h, 1), batchsize), lm.LayerDropProb),
        lm.lstm_layer2,
        VarDrop((size(lm.lstm_layer2.layer.cell.h, 1), batchsize), lm.LayerDropProb),
        lm.lstm_layer3,
        VarDrop((size(lm.lstm_layer3.layer.cell.h, 1), batchsize), lm.FinalDropProb),
        x -> tiedEmbeddings(x, lm.embedding_layer),
        softmax
    )

    # Fine-Tuning loops
    for epoch=1:epochs
        println("\nEpoch: $epoch")
        for i=1:num_of_iters

            # FORWARD
            l, H_prev = forward(model_layers, take!(gen), lm, batchsize, H_prev, α, β)

            # Slanted triangular learning rate step
            cut = num_of_iters * stlr_cut_frac
            peak = (i < cut) ? i/cut : (1 - ((i-cut)/(cut*(1/stlr_cut_frac-1))))
            ηL = 0.01*((1+peak*(stlr_ratio-1))/stlr_ratio)

            # Backprop with discriminative fine-tuning step
            discriminative_step!(model_layers[[1, 3, 5, 7]], ηL, l, gradient_clip)

            # ASGD Step, after Triggering
            asgd_step!.(i, [lm.lstm_layer1,lm.lstm_layer2,lm.lstm_layer3])

            println("loss: $l", " iteration number: $i")

            # Saving checkpoints
            if i == checkpoint_iter save_model!(lm) end
        end
    end
end
