"""
ULMFiT - FINE-TUNING
"""

using Flux
using Flux: Tracker
# using CorpusLoaders

cd(@__DIR__)
include("pretrain_lm.jl")    # importing LanguageModel and useful functions
include("custom_layers.jl")      # importing AWD_LSTM, VarDrop and DroppedEmbeddings
include("utils.jl")         # importing utilities

function discriminative_step!(layers, ηL::Float64, l, gradient_clip::Float64, opts::Vector)
    # Applying gradient clipping
    l = Tracker.hook(x -> grad_clipping(x, gradient_clip), l)

    # Gradient calculation
    grads = Tracker.gradient(() -> l, get_trainable_params(layers))

    # discriminative step
    ηl = ηL/(2.6^(length(layers)-1))
    for (layer, opt) in zip(layers, opts)
        opt.eta = ηl
        for ps in get_trainable_params([layer])
            Tracker.update!(opt, ps, grads[ps])
        end
        ηl *= 2.6
    end
    return
end

# Fine-Tuning Language Model
function fine_tune_lm!(lm::LanguageModel; batchsize::Integer=64, bptt::Integer=70, gradient_clip::Float64=0.25,
        ηL::Float64=4e-3, stlr_cut_frac::Float64=0.1, stlr_ratio::Float32=32, stlr_η_max::Float64=0.01, epochs::Integer=1, checkpoint_itvl::Integer=5000)

    model_layers = Chain(
        lm.embedding_layer,
        VarDrop(lm.wordDropProb),
        lm.lstm_layer1,
        VarDrop(lm.LayerDropProb),
        lm.lstm_layer2,
        VarDrop(lm.LayerDropProb),
        lm.lstm_layer3,
        VarDrop(lm.FinalDropProb),
        x -> lm.embedding_layer(x, true),
        softmax
    )

    opts = [ADAM(0.001, (0.7, 0.99)) for i=1:4]
    gpu!.(model_layers)

    # Fine-Tuning loops
    for epoch=1:epochs
        gen = Channel(x -> generator(x, imdb_fine_tune_data(); batchsize=batchsize, bptt=bptt))
        num_of_iters = take!(gen)
        T = num_of_iters-Int(floor((num_of_iters*2)/100))
        set_trigger!.(T, model_layers[[3, 5, 7]])
        cut = num_of_iters * stlr_cut_frac
        for i=1:num_of_iters

            # FORWARD
            l = loss(lm, model_layers, gen)

            # Slanted triangular learning rate step
            p_frac = (i < cut) ? i/cut : (1 - ((i-cut)/(cut*(1/stlr_cut_frac-1))))
            ηL = stlr_η_max*((1+p_frac*(stlr_ratio-1))/stlr_ratio)

            # Backprop with discriminative fine-tuning step
            discriminative_step!(model_layers[[1, 3, 5, 7]], ηL, l, gradient_clip, opts)

            # ASGD Step, after Triggering
            asgd_step!.(i, [lm.lstm_layer1,lm.lstm_layer2,lm.lstm_layer3])

            # Resetting dropout masks for all the layers with DropOut or DropConnect
            reset_masks!.(model_layers)

            println("loss: $l", " iteration completed: $i")

            # Saving checkpoints
            if i == checkpoint_itvl save_model!(lm) end
        end
        println("\nEpoch: $epoch")
    end
end
