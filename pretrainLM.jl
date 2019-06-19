"""
ULMFiT - LANGUAGE MODEL [Word-by-Word]
"""

using WordTokenizers   #For accesories
using InternedStrings   #For using Interned strings
using DelimitedFiles   # For reading and writing files
using Flux  #For building models
using Flux: Tracker, crossentropy, chunk
using LinearAlgebra: norm
using BSON: @save, @load  ##For saving model weights
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

    function LanguageModel(;embedDropProb::Float64 = 0.05, wordDropProb::Float64 = 0.4, hidDropProb::Float64 = 0.5, LayerDropProb::Float64 = 0.3, FinalDropProb::Float64 = 0.4)
        lm = new(
            intern.(string.(readdlm("vocab.csv",',', header=false)[:, 1])),
            gpu(LSTM(400, 1150; init = init_weights)),
            gpu(LSTM(1150, 1150; init = init_weights)),
            gpu(LSTM(1150, 400; init = init_weights)),
            embedDropProb,
            wordDropProb,
            hidDropProb,
            LayerDropProb,
            FinalDropProb
        )
        lm.vocab[1] = "<unk>"; lm.vocab[2] = "<pos>";push!(lm.vocab, "<eos>")  ## TO be implemented in datadep file
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

onehot(wordVect::Vector, vocab::Vector) =
    oh_repr = broadcast(x -> (x ∈ lm.vocab) ? Flux.onehot(x, vocab) : Flux.onehot("<unk>", vocab), wordVect)

function embeddingDropout(lm::LanguageModel)
    dropoutMask = rand(size(lm.embedMat)[1], 1) .> lm.embedDropProb
    return lm.embedMat .* gpu(dropoutMask)
end

embeddings(ohVect::Flux.OneHotMatrix, emDropMat::TrackedArray) = transpose(emDropMat)*ohVect

TiedEmbeddings(in::TrackedArray, emDropMat::TrackedArray) = emDropMat*in

# Mask generation
dropMask(p, shape; type = Float64) = (rand(type, shape...) .> p) .* type(1/(1 - p))

# Apply dropping
drop(x, mask) = x .* gpu(mask)

# DropConnect for lstm Layers
function dropConnect(lm::LanguageModel)
    droppedWeights = Dict{String, Vector{Array{Float32,2}}}([("Wi", []), ("Wh", [])])
    for layer in [lm.lstmLayer1, lm.lstmLayer2, lm.lstmLayer3]
        maskWi = dropMask(lm.hidDropProb, size(layer.cell.Wi); type = Float32)
        maskWh = dropMask(lm.hidDropProb, size(layer.cell.Wh); type = Float32)
        push!(droppedWeights["Wi"], copy(layer.cell.Wi.data))
        push!(droppedWeights["Wh"], copy(layer.cell.Wh.data))
        layer.cell.Wi.data .= drop(layer.cell.Wi.data, maskWi)
        layer.cell.Wh.data .= drop(layer.cell.Wh.data, maskWh)
    end
    return droppedWeights
end

# To restore dropped weight during DropConnect
function restoreWeights!(lm::LanguageModel, droppedWeights)
    lstms = [lm.lstmLayer1, lm.lstmLayer2, lm.lstmLayer3]
    for num=1:3
        lstms[num].cell.Wi.data .= gpu(droppedWeights["Wi"][num])
        lstms[num].cell.Wh.data .= gpu(droppedWeights["Wh"][num])
    end
    return nothing
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
        x -> embeddings(x, emDropMat),
        x -> drop(x, masks["wordDropMask"]),
        lm.lstmLayer1,
        x -> drop(x, masks["layerDropMask"]),
        lm.lstmLayer2,
        x -> drop(x, masks["layerDropMask"]),
        lm.lstmLayer3,
        x -> drop(x, masks["finalDropMask"]),
        x -> TiedEmbeddings(x, emDropMat)
    ).(X)

    restoreWeights!(lm, droppedWeights)
    return X
end

# objective funciton
function loss(H, Y)
    l = sum(crossentropy.(H, Y))
    Flux.truncate!(lm)
    return l
end

function fit!(lm::LanguageModel; batchsize::Integer=70, bptt::Integer=70,
    gradient_clip::Float64=0.25, initLearnRate::Number=30, epochs::Integer=1, α::Number=2, β::Number=1)

    corpus = loadCorpus()
    gen = Channel(x -> generator(x, corpus; batchsize = batchsize, bptt = bptt))

    opt = Descent(initLearnRate)    # Optimizer

    num_of_batches = take!(gen) # Number of mini-batches
    T = Int(floor((num_of_batches*2)/100))   # Averaging Trigger
    p = params(lm)
    Ht_prev = gpu.(repeat([zeros(Float32, length(lm.vocab), batchsize)], bptt))

    for epoch=1:epochs
        for i=1:num_of_batches

            # FORWARD
            X, Y = take!(gen)
            X, Y = gpu.(broadcast(x -> hcat(onehot(x, lm.vocab)...), X)), gpu.(broadcast(y -> hcat(onehot(y, lm.vocab)...), Y))
            Ht = softmax.(forward(X, lm))

            # BACKWARD
            # Loss calculation with AR and TAR regulatization
            l = loss(Ht, Y) + α*sum(norm, cpu.(Ht)) + β*sum(norm, cpu.(Ht .- Ht_prev))
            grads = Tracker.gradient(() -> l, p)
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
            if ((i/num_of_batches)*100)%5 == 0
                save_model!(lm)
            end
        end
    end
    println("\nEpoch: $epoch")
end

# To save model
function save_model!(lm::LanguageModel, filepath::String="ULMFiT-LM.bson")
    weights = Tracker.data.(params(lm))
    @save filepath weights
end
# To load model
function load_model!(lm::LanguageModel, filepath::String="ULMFiT-LM.bson")
    @load filepath weights
    Flux.loadparams!(lm, weights)
end
