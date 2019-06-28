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
using CuArrays  # For GPU support

cd(@__DIR__)

# import AWD_LSTM layer
include("awd_lstm.jl")

# including WikiText-103 Corpus
include("WikiText103_DataDeps.jl")

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
            gpu(AWD_LSTM(inLSTMSize, hidLSTMSize, hidDropProb; init = (dims..) -> init_weights(1/hidLSTMSize, dims...))),
            gpu(AWD_LSTM(hidLSTMSize, hidLSTMSize, hidDropProb; init = (dims..) -> init_weights(1/hidLSTMSize, dims...))),
            gpu(AWD_LSTM(hidLSTMSize, outLSTMSize, hidDropProb; init = (dims..) -> init_weights(1/hidLSTMSize, dims...))),
            embedDropProb,
            wordDropProb,
            LayerDropProb,
            FinalDropProb
        )
        lm.embedding_layer = gpu(DroppedEmbeddings(length(lm.vocab), 400, 0.1, true); init = (dims..) -> init_weights(0.1, dims...))
        return lm
    end;
end

Flux.@treelike LanguageModel

# Generator, whenever it is called it gives one mini-batch
function generator(c::Channel, corpus; batchsize::Integer=70, bptt::Integer=70)
    X_total, n = padding(chunk(corpus, batchsize))
    put!(c, n)
    for i=1:Int(floor(n/bptt))
        start = bptt*(i-1) + 1
        batch = [Flux.batch(X_total[k][j] for k=1:batchsize) for j=start:start+bptt]
        put!(c, batch[1:end-1])
        put!(c, batch[2:end])
    end
end

# Forward pass
function forward(model_layers, batch::AbstractVector, lm::LanguageModel, batchsize::Integer, H_prev::AbstractArray, α, β)
    batch = broadcast(x -> hcat(onehot(x, lm.vocab)...), batch)
    resetMasks!.(model_layers[1:end-2])
    batch = broadcast(x -> model_layers(gpu(x)), batch)

    Y = take!(gen)
    Y = broadcast(y -> gpu(hcat(onehot(y, lm.vocab)...)), Y)
    l = loss(batch, Y, gpu.(H_prev), α, β)

    return l, Tracker.data.(batch)
end

# loss funciton - Loss calculation with AR and TAR regulatization
function loss(H, Y, H_prev, α, β)
    expr(ht, yt, ht_prev) = sum(crossentropy(ht, yt)) + α*sum(norm, ht) + β*sum(norm, ht .- ht_prev)
    l = sum(broadcast((ht, yt, ht_prev) -> expr(ht, yt, ht_prev), batch, Y, H_prev))
    Flux.truncate!(lm)
    return l, Tracker.data.(H)
end

# Backward
function back!(p::Flux.Params, l, opt, gradient_clip::Float64)
    grads = Tracker.gradient(() -> l, p)
    for ps in p     # Applying Gradient clipping
        grads[ps].data = min.(grads[ps].data, gradient_clip)
    end
    Tracker.update!(opt, p, grads)
    return nothing
end

# Funciton for training Language Model
function fit!(lm::LanguageModel; batchsize::Integer=70, bptt::Integer=70,gradient_clip::Float64=0.25,
        initLearnRate::Number=30, epochs::Integer=1, α::Number=2, β::Number=1, checkpointIter::Integer=)

    # Initializations
    gen = Channel(x -> generator(x, loadCorpus(); batchsize = batchsize, bptt = bptt))
    opt = Descent(initLearnRate)    # Optimizer
    num_of_batches = take!(gen) # Number of mini-batches
    T = Int(floor((num_of_batches*2)/100))   # Averaging Trigger
    p = params(lm)
    H_prev = repeat([zeros(Float32, length(lm.vocab), batchsize)], bptt)

    model_layers = Chain(
        lm.embedding_layer,
        VarDrop(lm.wordDropProb, (400, batchsize)),
        lm.lstm_layer1,
        VarDrop(lm.LayerDropProb, (1150, batchsize)),
        lm.lstm_layer2,
        VarDrop(lm.LayerDropProb, (1150, batchsize)),
        lm.lstm_layer3,
        VarDrop(lm.FinalDropProb, (400, batchsize)),
        x -> tiedEmbeddings(x, lm.embedding_layer),
        softmax
    )

    # Pre-Training loops
    for epoch=1:epochs
        println("\nEpoch: $epoch")
        for i=1:num_of_batches

            # FORWARD
            l, H_prev = forward(model_layers, take!(gen), lm, batchsize, H_prev, α, β)

            # BACKWARD
            back!(p, l, opt, gradient_clip)

            #ASGD Step
            asgd_step!.(i, [lm.lstm_layer1,lm.lstm_layer2,lm.lstm_layer3])

            println("loss: $l", " iteration number: $i")

            # Saving checkpoints
            if i == checkpointIter save_model!(lm) end
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

# Sampling
function sampling(lm::LanguageModel, startingText::String)
    LSTMs = Chain(
        lm.lstm_layer1,
        lm.lstm_layer2,
        lm.lstm_layer3,
    )

    tokens = tokenize(startingText)
    ohrep = onehot(tokens, lm.vocab)
    embeddings = map(x -> embeddingLayer(x, lm.embedMat), ohrep)
    h = (LSTMs.(embeddings))[end]
    probabilities = softmax(tiedDecoder(h, lm.embedMat))
    prediction = lm.vocab[findall(isequal(maximum(probabilities)), probabilities)]
    print(prediction[1], ' ')

    while true
        h = LSTMs(h)
        probabilities = softmax(tiedDecoder(h, lm.embedMat))
        prediction = lm.vocab[findall(isequal(maximum(probabilities)), probabilities)]
        print(prediction[1], ' ')
        prediction == "_pad_" && break
    end
    println(prediction)
end
