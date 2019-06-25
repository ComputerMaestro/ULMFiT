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

# Initializing funciton for model LSTM weights
init_weights(dims...) = randn(Float32, dims...) .* sqrt(Float32(1/1150))

# Language Model
mutable struct LanguageModel
    vocab :: Vector
    lstmLayer1 :: Flux.Recur
    lstmLayer2 :: Flux.Recur
    lstmLayer3 :: Flux.Recur
    embedDropProb :: Float64
    wordDropProb :: Float64
    hidDropProb :: Float64
    LayerDropProb :: Float64
    FinalDropProb :: Float64
    embedMat :: TrackedArray

    function LanguageModel(inLSTMSize::Integer=400, hidLSTMSize::Integer=1150, outLSTMSize::Integer=inLSTMSize;
        embedDropProb::Float64 = 0.05, wordDropProb::Float64 = 0.4, hidDropProb::Float64 = 0.5, LayerDropProb::Float64 = 0.3, FinalDropProb::Float64 = 0.4)
        lm = new(
            intern.(string.(readdlm("vocab.csv",',', header=false)[:, 1])),
            gpu(LSTM(inLSTMSize, hidLSTMSize; init = init_weights)),
            gpu(LSTM(hidLSTMSize, hidLSTMSize; init = init_weights)),
            gpu(LSTM(hidLSTMSize, outLSTMSize; init = init_weights)),
            embedDropProb,
            wordDropProb,
            hidDropProb,
            LayerDropProb,
            FinalDropProb
        )
        lm.embedMat = gpu(param(randn(Float32, size(lm.vocab)[1], 400) .* 0.1f0))
        return lm
    end
end

Flux.@treelike LanguageModel

# Loading corpus and preprocessing steps
cd(@__DIR__)
include("WikiText103_DataDeps.jl")

# Loading Corpus
function loadCorpus(corpuspath::String = joinpath(datadep"WikiText-103", "wiki.train.tokens"))
    corpus = read(open(corpuspath, "r"), String)
    return intern.(tokenize(corpus))
end

#Adding "<pad>" keyowrd at the end if the length of the sentence is < bptt
function padding(batches::Vector)
    n = maximum([length(x) for x in batches])
    return ([length(batch) < n ? cat(batch, repeat(["<pos>"], n-length(batch)); dims = 1) : batch[1:n] for batch in batches], n)
end

# Generator, whenever it is called it gives one mini-batch
function generator(c::Channel, corpus; batchsize::Integer=70, bptt::Integer=70)
    X_total, n = padding(chunk(corpus, batchsize))
    put!(c, n)
    for i=1:Int(floor(n/bptt))
        start = bptt*(i-1) + 1
        batch = [Flux.batch(X_total[k][j] for k=1:batchsize) for j=start:start+bptt]
        put!(c, (batch[1:end-1], batch[2:end]))
    end
end

# Converts vector of words to vector of one-hot vectors
onehot(wordVect::Vector, vocab::Vector) =
    oh_repr = broadcast(x -> (x ∈ vocab) ? Flux.onehot(x, vocab) : Flux.onehot("_unk_", vocab), wordVect)

# Gives dropped out embeddings matrix
embeddingDropout(lm::LanguageModel) = lm.embedMat .* (gpu(rand(size(lm.embedMat)[1], 1)) .> lm.embedDropProb)

# Converitng one-hot matrix to word embedddings using dropped embedding matrix
embeddingLayer(ohrepr::Union{Flux.OneHotMatrix, Flux.OneHotVector}, emDropMat::TrackedArray) = transpose(emDropMat)*ohrepr

# Weight-tying of embedding layer with softmax layer
tiedDecoder(in::TrackedArray, emDropMat::TrackedArray) = emDropMat*in

# Mask generation
dropMask(p, shape; type = Float32) = gpu(rand(type, shape...) .> p) .* type(1/(1 - p))

# Apply dropping
struct VarDrop
    mask::AbstractArray
end

VarDrop(p, shape; type = Float32) = VarDrop(dropMask(p, shape; type = Float32))

(d::VarDrop)(in) = in .* d.mask

# Drop Connect
struct DropConnect
    droppedWi::Array{Float32, 2}
    droppedWh::Array{Float32, 2}
    layer
end

function DropConnect(p, layer)
    dc = DropConnect(copy(layer.cell.Wi.data), copy(layer.cell.Wh.data), layer)
    maskWi = dropMask(lm.hidDropProb, size(layer.cell.Wi); type = Float32)
    maskWh = dropMask(lm.hidDropProb, size(layer.cell.Wh); type = Float32)
    layer.cell.Wi.data .= (layer.cell.Wi.data .* maskWi)
    layer.cell.Wh.data .= (layer.cell.Wh.data .* maskWh)
    return dc
end

(dc::DropConnect)(in) = dc.layer(in)

function restoreWeights!(dc::DropConnect)
    dc.layer.cell.Wi.data .= gpu(dc.droppedWi)
    dc.layer.cell.Wh.data .= gpu(dc.droppedWh)
    return nothing
end

# Forward pass
function forward(X, lm::LanguageModel, batchsize)
    emDropMat = embeddingDropout(lm)
    droppedWeights = dropConnect(lm)

    Layers = Chain(
        x -> embeddingLayer(x, emDropMat),
        VarDrop(lm.wordDropProb, (400, batchsize)),
        DropConnect(lm.lstmLayer1),
        VarDrop(lm.LayerDropProb, (1150, batchsize)),
        DropConnect(lm.lstmLayer2),
        VarDrop(lm.LayerDropProb, (1150, batchsize)),
        DropConnect(lm.lstmLayer3),
        VarDrop(lm.FinalDropProb, (400, batchsize)),
        x -> tiedDecoder(x, emDropMat)
    )
    X = broadcast(x -> cpu(Layers(gpu(x))), X)

    restoreWeights!.(Layers[[3, 5, 7]])
    return softmax.(X)
end

# objective funciton
function loss(H, Y)
    l = sum(crossentropy.(H, Y))
    Flux.truncate!(lm)
    return l
end

function fit!(lm::LanguageModel; batchsize::Integer=70, bptt::Integer=70,
    gradient_clip::Float64=0.25, initLearnRate::Number=30, epochs::Integer=1, α::Number=2, β::Number=1, checkpointIter::Integer=)

    corpus = loadCorpus()
    gen = Channel(x -> generator(x, corpus; batchsize = batchsize, bptt = bptt))

    opt = Descent(initLearnRate)    # Optimizer

    num_of_batches = take!(gen) # Number of mini-batches
    T = Int(floor((num_of_batches*2)/100))   # Averaging Trigger
    p = params(lm)
    Ht_prev = gpu.(repeat([zeros(Float32, length(lm.vocab), batchsize)], bptt))

    for epoch=1:epochs
        println("\nEpoch: $epoch")
        for i=1:num_of_batches

            # FORWARD
            X, Y = take!(gen)
            X, Y = broadcast(x -> hcat(onehot(x, lm.vocab)...), X), broadcast(y -> hcat(onehot(y, lm.vocab)...), Y)
            Ht = forward(X, lm, batchsize)

            # BACKWARD
            # Loss calculation with AR and TAR regulatization
            l = loss(Ht, Y) + α*sum(norm, cpu.(Ht)) + β*sum(norm, cpu.(Ht .- Ht_prev))
            grads = Tracker.gradient(() -> l, p)
            for ps in p     # Applying Gradient clipping
                grads[ps].data = min.(grads[ps].data, gradient_clip)
            end
            Tracker.update!(opt, p, grads)
            Ht_prev = [h.data for h in Ht]

            # ASGD Step
            if i >= T
                avg_fact = 1/max(i - T + 1, 1)
                if avg_fact != 1
                    accumulator = accumulator + Tracker.data.(p)
                    i = 1
                    for ps in p
                        ps.data .= avg_fact*copy(accumulator[i])
                        i += 1
                    end
                else
                    accumulator = deepcopy(Tracker.data.(p))   # Accumulator for ASGD
                end
            end

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
        lm.lstmLayer1,
        lm.lstmLayer2,
        lm.lstmLayer3,
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
