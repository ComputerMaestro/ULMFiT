"""
ULMFiT - LANGUAGE MODEL [Word-by-word]
"""

using WordTokenizers   #For accesories
using InternedStrings   #For using Interned strings
using DelimitedFiles   # For reading and writing files
using BSON  ##For saving model weights
using Flux  #For building models
using Flux: Tracker, onecold, crossentropy, chunk, batch
using Base.Iterators: partition
using LinearAlgebra: norm

# Initializing wegiths between [-1/sqrt(H), 1/sqrt(H)] where H = hidden size
init_weights(dims...) = randn(Float32, dims...) .* sqrt(Float32(1/1150))

mutable struct LanguageModel
    vocab :: Vector
    lstmLayer1 :: Flux.Recur
    lstmLayer2 :: Flux.Recur
    lstmLayer3 :: Flux.Recur
    embedDropProb :: Float64
    wordDropout :: Dropout
    hiddenDropout :: Dropout
    LayerDropProb :: Float64
    FinalDropProb :: Float64
    embedMat :: TrackedArray
    RecurrentLayers :: Chain

    function LanguageModel(;embedDropProb::Float64 = 0.05, wordDropProb::Float64 = 0.4, hidDropProb::Float64 = 0.5, LayerDropProb::Float64 = 0.3, FinalDropProb::Float64 = 0.4)
        lm = new(
            readdlm("vocab.csv",',', header=false)[:, 1],
            LSTM(400, 1150; init = init_weights),
            LSTM(1150, 1150; init = init_weights),
            LSTM(1150, 400; init = init_weights),
            embedDropProb,
            Dropout(wordDropProb),
            Dropout(hidDropProb),
            LayerDropProb,
            FinalDropProb
        )
        lm.vocab[1] = "<unk>"; lm.vocab[2] = "<pos>";push!(lm.vocab, "<eos>")  ## TO be implemented in datadep file
        lm.embedMat = param(randn(Float32, size(lm.vocab)[1], 400) .* 0.1f0)
        lm.RecurrentLayers = Chain(
            lm.lstmLayer1,
            Dropout(LayerDropProb),
            lm.lstmLayer2,
            Dropout(LayerDropProb),
            lm.lstmLayer3,
            Dropout(FinalDropProb)
        )

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

# function hiddenDropout(unit, lm :: LanguageModel)
#     unit.cell.Wh = lm.hiddenDropout(unit.cell.Wh)
#     unit.cell.Wi = lm.hiddenDropout(unit.cell.Wi)
#     return unit
# end

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

#Adding "<pad>" keyowrd at the end if the length of the sentence is < bptt
function padding(batches::Vector)
    n = maximum([length(x) for x in batches])
    return ([length(batch) < n ? cat(batch, repeat(["<pos>"], n-length(batch)); dims = 1) : batch[1:n] for batch in batches], n)
end

# Generator, whenever it is called it gives one mini-batch
function generator(c::Channel, corpus = corpus; batchsize::Integer=70, bptt::Integer=70)
    X_total, n = padding(chunk(corpus, batchsize))
    X_total = [batch(X_total[j][i] for j=1:length(X_total)) for i=1:n]
    Y_total = collect(partition(X_total[2:end], bptt))
    X_total = collect(partition(X_total[1:end-1], bptt))
    for data in zip(X_total, Y_total)
        put!(c, data)
    end
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
        X, Y = take!(gen)
        X, Y = broadcast(x -> onehot(x, lm.vocab), X), broadcast(y -> hcat(onehot(y, lm.vocab)...), Y)

        # FORWARD
        emDropMat = embeddingDropout(lm)
        X = broadcast(x -> embeddings(x, emDropMat, lm), X)    #vector of matrices of embeddings of input sentences
        X = lm.wordDropout.(X)
        X = lm.RecurrentLayers.(X) #outputMat is a vector of matrices of output sentences
        X = broadcast(x -> TiedEmbeddings(x, emDropMat), X)
        Ht = softmax.(X)
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
