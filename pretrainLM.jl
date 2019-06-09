"""
ULMFiT - LANGUAGE MODEL [Word-by-Word]
"""

using WordTokenizers   #For accesories
using InternedStrings   #For using Interned strings
using DelimitedFiles   # For reading and writing files
using BSON  ##For saving model weights
using Flux  #For building models
using Flux: Tracker, onecold, crossentropy, chunk, batch
using LinearAlgebra: norm

init_weights(dims...) = randn(Float32, dims...) .* sqrt(Float32(1/1150))

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

    function LanguageModel(;embedDropProb::Float64 = 0.05, wordDropProb::Float64 = 0.4, hidDropProb::Float64 = 0.5, LayerDropProb::Float64 = 0.3, FinalDropProb::Float64 = 0.4)
        lm = new(
            readdlm("vocab.csv",',', header=false)[:, 1],
            LSTM(400, 1150; init = init_weights),
            LSTM(1150, 1150; init = init_weights),
            LSTM(1150, 400; init = init_weights),
            embedDropProb,
            wordDropProb,
            hidDropProb,
            LayerDropProb,
            FinalDropProb
        )
        lm.vocab[1] = "<unk>"; lm.vocab[2] = "<pos>";push!(lm.vocab, "<eos>")  ## TO be implemented in datadep file
        lm.embedMat = param(randn(Float32, size(lm.vocab)[1], 400) .* 0.1f0)
        return lm
    end
end

Flux.@treelike LanguageModel

#Loading corpus and preprocessing steps
cd(@__DIR__)
include("WikiText103_DataDeps.jl")

function loadCorpus(corpuspath::String = joinpath(datadep"WikiText-103", "wiki.train.tokens"))
    corpus = read(open(corpuspath, "r"), String)
    return tokenize(corpus)
end

#Adding "<pad>" keyowrd at the end if the length of the sentence is < bptt
function padding(batches::Vector)
    n = maximum([length(x) for x in batches])
    return ([length(batch) < n ? cat(batch, repeat(["<pos>"], n-length(batch)); dims = 1) : batch[1:n] for batch in batches], n)
end

# Generator, whenever it is called it gives one mini-batch
function generator(c::Channel, corpus; batchsize::Integer=70, bptt::Integer=70)
    X_total, n = padding(chunk(corpus, batchsize))
    for i=1:Int(floor(n/bptt))
        start = bptt*(i-1) + 1
        batch = [Flux.batch(X_total[k][j] for k=1:batchsize) for j=start:start+bptt]
        put!(c, (batch[1:end-1], batch[2:end]))
    end
end

onehot(wordVect::Vector, vocab::Vector) =
    oh_repr = broadcast(x -> (x ∈ lm.vocab) ? Flux.onehot(x, vocab) : Flux.onehot("<unk>", vocab), wordVect)

function embeddingDropout(lm :: LanguageModel)
    dropoutMask = rand(size(lm.embedMat)[1], 1) .> lm.embedDropProb
    return lm.embedMat .* dropoutMask
end

function embeddings(ohVect::Vector{Flux.OneHotVector}, emDropMat::TrackedArray, lm::LanguageModel)
    embeddings = Array{Bool, 2}(UndefInitializer(), length(lm.vocab), 0)
    embeddings = hcat(embeddings, hcat(ohVect...))
    return transpose(emDropMat)*embeddings
end

TiedEmbeddings(in::TrackedArray, emDropMat::TrackedArray) = emDropMat*in

# Mask generation
dropMask(p, shape; type = Float64) = (rand(type, shape...) .> p) .* type(1/(1 - p))

# Apply dropout
dropout(x, mask) = x .* mask

# DropConnect for lstm Layers
function dropConnect(lm::LanguageModel)
    droppedWeights = Dict{String, Vector{Array{Float32,2}}}([("Wi", []), ("Wh", [])])
    for layer in [lm.lstmLayer1, lm.lstmLayer2, lm.lstmLayer3]
        maskWi = dropMask(lm.hidDropProb, size(layer.cell.Wi); type = Float32)
        maskWh = dropMask(lm.hidDropProb, size(layer.cell.Wh); type = Float32)
        push!(droppedWeights["Wi"], layer.cell.Wi.data)
        push!(droppedWeights["Wh"], layer.cell.Wh.data)
        layer.cell.Wi = layer.cell.Wi .* maskWi
        layer.cell.Wh = layer.cell.Wh .* maskWh
    end
    return droppedWeights
end

# To restore dropped weight during DropConnect
function restoreWeights!(lm::LanguageModel, droppedWeights)
    lstms = [lm.lstmLayer1, lm.lstmLayer2, lm.lstmLayer3]
    for num=1:3
        lstms[num].cell.Wi.data .= droppedWeights["Wi"][num]
        lstms[num].cell.Wh.data .= droppedWeights["Wh"][num]
    end
    return lm
end

# Forward pass
function forward(X, lm::LanguageModel)
    emDropMat = embeddingDropout(lm)
    masks = Dict()
    masks["wordDropMask"] = dropMask(lm.wordDropProb, (400, batchsize))
    masks["layerDropMask"] = dropMask(lm.LayerDropProb, (1150, batchsize))
    masks["finalDropMask"] = dropMask(lm.FinalDropProb, (400, batchsize))

    droppedWeights = dropConnect(lm)

    X = Chain(
        x -> embeddings(x, emDropMat, lm),
        x -> dropout(x, masks["wordDropMask"]),
        lm.lstmLayer1,
        x -> dropout(x, masks["layerDropMask"]),
        lm.lstmLayer2,
        x -> dropout(x, masks["layerDropMask"]),
        lm.lstmLayer3,
        x -> dropout(x, masks["finalDropMask"]),
        x -> TiedEmbeddings(x, emDropMat)
    ).(X)

    restoreWeights!(lm, droppedWeights)
    return X
end

# objective funciton
function loss(X, Y)
    l = sum(crossentropy.(H, Y))
    Flux.truncate!(lm)
    return l
end

function fit!(lm::LanguageModel; batchsize::Integer=70, bptt::Integer=70,
    gradient_clip::Float64=0.25, initLearnRate::Number=30; epochs::Integer=1, α::Number=2, β::Number=1)

    corpus = loadCorpus()
    gen = Channel(x -> generator(x, corpus; batchsize = batchsize, bptt = bptt))
    opt = Descent(initLearnRate)
    T = Int(floot((K*2)/100))   # Averaging Trigger
    accumulator = Dict(); k = 0
    for epoch=1:epochs

        # FORWARD
        X, Y = take!(gen)
        X, Y = broadcast(x -> onehot(x, lm.vocab), X), broadcast(y -> hcat(onehot(y, lm.vocab)...), Y)
        Ht = softmax.(forward(X, lm))
        Ht_prev = [h.data for h in Ht]

        # BACKWARD
        p = params(lm)

        #Loss calculation with AR and TAR regulatization
        l = loss(Ht, Y) + α*sum(norm, Ht) + β*sum(norm, Ht .- Ht_prev)
        grads = Tracker.gradient(() -> l, p)
        update!(opt, p, grads)

        # ASGD Step
        k += 1
        avg_fact = 1/max(k - T, 1)
        if avg_fact != 1
            for ps in p
                accumulator[ps] = accumulator[ps] .+ ps.data
                ps.data = avg_fact*(accumulator[ps])
            end
        end
    end
end
