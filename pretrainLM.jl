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
include("awd_lstm.jl")      # importing AWD_LSTM (ASGD Weight-Dropped LSTM)
include("utils.jl")     # including some functions
include("WikiText103_DataDeps.jl")      # including WikiText-103 Corpus

# Loading Corpus
function loadCorpus(corpuspath::String = joinpath(datadep"WikiText-103", "wiki.train.tokens"))
    corpus = read(open(corpuspath, "r"), String)
    return intern.(tokenize(corpus))
end

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

    function LanguageModel(inLSTMSize::Integer=400, hidLSTMSize::Integer=1150, outLSTMSize::Integer=inLSTMSize;
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
        lm.embedding_layer = DroppedEmbeddings(length(lm.vocab), 400, 0.1; init = (dims...) -> init_weights(0.1, dims...))
        return lm
    end
end

Flux.@treelike LanguageModel

# Generator, whenever it is called it gives one mini-batch
function generator(c::Channel, corpus; batchsize::Integer=64, bptt::Integer=70)
    X_total, n = padding(chunk(corpus, batchsize))
    put!(c, n)
    for i=1:Int(floor(n/bptt))
        start = bptt*(i-1) + 1
        batch = [Flux.batch(X_total[k][j] for k=1:batchsize) for j=start:start+bptt]
        put!(c, batch[1:end-1])
        put!(c, batch[2:end])
    end
end

function forward(model_layers, x, lm, h_prev, α, β, ar_value, tar_value)
    x = gpu(model_layers[1](indices(x, lm.vocab, "_unk_")))
    x = model_layers[2:7](x)
    h_prev == nothing || (tar_value += β*sum(norm, h_prev .- x))    #calculates TAR value (regularization)
    h_prev = copy(x)
    x = model_layers[8](x)
    ar_value += α*sum(norm, x)  # Calculates AR value (regularization)
    x = model_layers[9:end](x)
    return x, h_prev, ar_value, tar_value
end

# loss funciton - Loss calculation with AR and TAR regulatization
function loss(X, Y, lm)
    l, ar_value, tar_value = 0.0f0, 0.0f0, 0.0f0
    h_prev = nothing
    for x, y in zip(X, Y)
        x, h_prev, ar_value, tar_value = forward(model_layers, x, lm, h_prev, α, β, ar_value, tar_value)
        y = indices(y, lm.vocab, "_unk_")
        y = gpu(Flux.onehotbatch(y, lm.vocab, "_unk_"))
        l += sum(crossentropy(x, y))
    end
    Flux.truncate!(lm)
    return l+ar_value+tar_value
end

# Gradient Clipping
grad_clipping(g, upper_bound) = min(g, upper_bound)

# Backward
function back!(p::Flux.Params, l, opt, gradient_clip::Float64)
    l = Tracker.hook(x -> grad_clipping(x, gradient_clip), l)
    grads = Tracker.gradient(() -> l, p)
    Tracker.update!(opt, p, grads)
    return
end

# Funciton for training Language Model
function fit!(lm::LanguageModel; batchsize::Integer=64, bptt::Integer=70, gradient_clip::Float64=0.25,
    base_lr=0.004, epochs::Integer=1, α::Number=2, β::Number=1, checkpoint_iter::Integer=5000)

    # All important layers for training
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

    # Initializations
    gen = Channel(x -> generator(x, loadCorpus(); batchsize = batchsize, bptt = bptt))
    opt = ADAM(base_lr, (0.7, 0.99))    # ADAM Optimizer
    num_of_batches = take!(gen) # Number of mini-batches
    T = Int(floor((num_of_batches*2)/100))   # Averaging Trigger
    set_trigger!.(T, model_layers[[3, 5, 7]])  # Setting triggers for AWD_LSTM layers
    p = params(lm)

    # Pre-Training loops
    for epoch=1:epochs
        println("\nEpoch: $epoch")
        for i=1:num_of_batches

            # FORWARD PROPAGATION
            X = take!(gen)
            Y = take!(gen)
            for layer in model_layers[1:8]
                gpu!(layer)
            end
            l = loss(X, Y, lm)

            # BACK PROPAGATION
            back!(p, l, opt, gradient_clip)

            # ASGD Step, after Triggering
            asgd_step!.(i, [lm.lstm_layer1,lm.lstm_layer2,lm.lstm_layer3])

            println("loss: $l", " iteration number: $i")

            # Saving checkpoints
            if i == checkpoint_iter save_model!(lm) end
        end
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
    return lm
end

# Converts parameters and masks to CuArrays for GPU support
function gpu!(lm::LanguageModel)
    lm.embedding_layer = gpu(lm.embedding_layer)
    lm.lstm_layer1 = gpu!(lm.lstm_layer1)
    lm.lstm_layer2 = gpu!(lm.lstm_layer2)
    lm.lstm_layer3 = gpu!(lm.lstm_layer3)
    return
end

function cpu!(lm::LanguageModel)
    lm.embedding_layer = cpu!(lm.embedding_layer)
    lm.lstm_layer1 = cpu!(lm.lstm_layer1)
    lm.lstm_layer2 = cpu!(lm.lstm_layer2)
    lm.lstm_layer3 = cpu!(lm.lstm_layer3)
    return
end

# Sampling
function sampling(starting_text::String, lm::LanguageModel=load_model!())
    model_layers = Chain(
        lm.lstm_layer1,
        lm.lstm_layer2,
        lm.lstm_layer3
    )

    reset_probability!.(0.0, model_layers)
    reset_masks!.(model_layers)

    tokens = tokenize(starting_text)
    word_indices = indices(tokens, lm.vocab, "_unk_")
    embeddings = lm.embedding_layer.(word_indices)
    h = (model_layers.(embeddings))[end]
    probabilities = softmax(lm.embedding_layer(h, true)))
    prediction = lm.vocab[findall(isequal(maximum(probabilities)), probabilities)[1]]
    println("SAMPLING...")
    print(prediction, ' ')

    while true
        h = model_layers(h)
        probabilities = softmax(lm.embedding_layer(h, true)))
        prediction = lm.vocab[findall(isequal(maximum(probabilities)), probabilities)[1]]
        print(prediction, ' ')
        prediction == "_pad_" && break
    end
    println(prediction)
end
