"""
ULMFiT - LANGUAGE MODEL [Word-by-Word]
"""

using WordTokenizers   # For accesories
using InternedStrings   # For using Interned strings
using DelimitedFiles   # For reading and writing files
using Flux  # For building models
using Flux: Tracker, crossentropy, chunk
using LinearAlgebra: norm
using BSON: @save, @load  # For saving model weights
# using CuArrays  # For GPU support

cd(@__DIR__)
include("custom_layers.jl")      # importing AWD_LSTM (ASGD Weight-Dropped LSTM)
include("utils.jl")     # including some functions
include("WikiText103_DataDeps.jl")      # including WikiText-103 Corpus

# Language Model
mutable struct LanguageModel
    vocab :: Vector
    lstm_layer1 :: AWD_LSTM
    lstm_layer2 :: AWD_LSTM
    lstm_layer3 :: AWD_LSTM
    embedDropProb :: Float64
    wordDropProb :: Float64
    LayerDropProb :: Float64
    FinalDropProb :: Float64
    embedding_layer :: DroppedEmbeddings

    function LanguageModel(inLSTMSize::Integer=400, hidLSTMSize::Integer=1150, outLSTMSize::Integer=inLSTMSize, embedding_size::Integer=400;
        embedDropProb::Float64 = 0.05, wordDropProb::Float64 = 0.4, hidDropProb::Float64 = 0.5, LayerDropProb::Float64 = 0.3, FinalDropProb::Float64 = 0.4)
        lm = new(
            intern.(string.(readdlm("vocab.csv",',', header=false)[:, 1])),
            AWD_LSTM(inLSTMSize, hidLSTMSize, hidDropProb; init = (dims...) -> init_weights(1/hidLSTMSize, dims...)),
            AWD_LSTM(hidLSTMSize, hidLSTMSize, hidDropProb; init = (dims...) -> init_weights(1/hidLSTMSize, dims...)),
            AWD_LSTM(hidLSTMSize, outLSTMSize, hidDropProb; init = (dims...) -> init_weights(1/hidLSTMSize, dims...)),
            embedDropProb,
            wordDropProb,
            LayerDropProb,
            FinalDropProb
        )
        lm.embedding_layer = DroppedEmbeddings(length(lm.vocab), embedding_size, 0.1; init = (dims...) -> init_weights(0.1, dims...))
        return lm
    end
end

Flux.@treelike LanguageModel

# Forward
function forward(lm, model_layers, batch)
    batch = broadcast(x -> gpu(model_layers[1](indices(x, lm.vocab, "_unk_"))), batch)
    batch = model_layers[2:end].(batch)
    return batch
end

# loss funciton - Loss calculation with AR and TAR regulatization
function loss(lm, model_layers, gen)
    H = forward(lm, model_layers, take!(gen))
    Y = broadcast(x -> Flux.onehotbatch(x, lm.vocab, "_unk_"), take!(gen))
    l = sum(crossentropy.(H, Y))
    Flux.truncate!(model_layers)
    return l
end

# Gradient Clipping
grad_clipping(g, upper_bound) = min(g, upper_bound)

# Backward
function back!(layers, l, opt, gradient_clip::Float64)
    # Applying gradient clipping
    l = Tracker.hook(x -> grad_clipping(x, gradient_clip), l)

    # Calulating gradients
    p = get_trainable_params(layers)
    grads = Tracker.gradient(() -> l, p
    Tracker.update!(opt, p, grads)
    return
end

# Funciton for training Language Model
function pretrain_lm!(lm::LanguageModel, model_layers=nothing; batchsize::Integer=64, bptt::Integer=70, gradient_clip::Float64=0.25,
    base_lr=0.004, epochs::Integer=1, checkpoint_iter::Integer=5000)

    # Chain of all important layers to pass from
    model_layers = Chain(
        lm.embedding_layer,
        VarDrop(lm.wordDropProb),
        lm.lstm_layer1.layer,
        VarDrop(lm.LayerDropProb),
        lm.lstm_layer2.layer,
        VarDrop(lm.LayerDropProb),
        lm.lstm_layer3.layer,
        VarDrop(lm.FinalDropProb),
        x -> lm.embedding_layer(x, true),
        softmax
    )

    # Initializations
    opt = ADAM(base_lr, (0.7, 0.99))    # ADAM Optimizer
    gpu!.(model_layers)

    # Pre-Training loops
    for epoch=1:epochs
        gen = Channel(x -> generator(x, loadCorpus(); batchsize = batchsize, bptt = bptt))
        num_of_batches = take!(gen) # Number of mini-batches
        T = num_of_iters-Int(floor((num_of_iters*2)/100))   # Averaging Trigger
        set_trigger!.(T, model_layers[[3, 5, 7]])  # Setting triggers for AWD_LSTM layers
        for i=1:num_of_batches

            # FORWARD PROPAGATION
            l = loss(lm, model_layers, gen)

            # BACK PROPAGATION
            back!(p, l, opt, gradient_clip)
            for layer in model_layers[1:8]
                cpu!(layer)
            end

            # ASGD Step, after Triggering
            asgd_step!.(i, [lm.lstm_layer1, lm.lstm_layer2, lm.lstm_layer3])

            reset_masks!.(model_layers)

            println("loss: $l", " iteration number: $i")

            # Saving checkpoints
            if i == checkpoint_iter save_model!(lm) end
        end
        println("\nEpoch: $epoch")
    end
end

# To save model
function save_model!(lm::LanguageModel, filepath::String="ULMFiT-LM.bson")
    weights = Tracker.data.(params(lm))
    @save filepath weights
end

# To load model
function load_model!(lm::LanguageModel, filepath::String=joinpath(datadep"pretrained-ULMFiT", "weights.bson"))
    @load filepath weights
    Flux.loadparams!(lm, weights)
end

function load_model!()
    lm = LanguageModel()
    load_model!(lm)
    return lm
end

function load_model!(filepath::String)
    lm = LanguageModel()
    load_model!(lm, filepath)
end
