"""
ULMFiT - LANGUAGE MODEL [Word-by-word]
"""

using WordTokenizers   #For accesories
using InternedStrings   #For using Interned strings
using DelimitedFiles   # For reading and writing files
using BSON  ##For saving model weights
using Flux  #For building models
using Flux: Tracker, onecold, crossentropy#, onehot

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
include("WikiText103_DataDeps.jl")

function preprocess(corpuspath::String = joinpath(datadep"WikiText-103", "wiki.train.tokens"))
    corpus = read(open(corpuspath, "r"), String)
    sentences = split(lowercase(corpus), '\n')
    deleteat!(sentences, findall(x -> isequal(x, "")||isequal(x, " ")||(isequal(x[1:2], " =")&&isequal(x[prevind(x, lastindex(x), 1):end], "= ")), sentences))
    sentences .*= "<eos>"
    return sentences
end

function embeddingDropout(lm :: LanguageModel)
    dropoutMask = rand(size(lm.embedMat)[1], 1) .> lm.embedDropProb
    return lm.embedMat .* dropoutMask
end

function hiddenDropout(unit, lm :: LanguageModel)
    unit.cell.Wh = lm.hiddenDropout(unit.cell.Wh)
    unit.cell.Wi = lm.hiddenDropout(unit.cell.Wi)
    return unit
end

onehot(wordVect::Vector, vocab::Vector) =
    oh_repr = broadcast(x -> (x ∈ lm.vocab) ? Flux.onehot(x, vocab) : Flux.onehot("<unk>", vocab), wordVect)

function embeddings(ohVect::Vector{Flux.OneHotVector}, emDropMat::TrackedArray, lm::LanguageModel)
    embeddings = Array{Bool, 2}(UndefInitializer(), length(lm.vocab), 0)
    embeddings = hcat(embeddings, hcat(ohVect...))
    return transpose(emDropMat)*embeddings
end

TiedEmbeddings(in::TrackedArray, emDropMat::TrackedArray) = emDropMat*in

#Adding "<pad>" keyowrd at the end if the length of the sentence is < bptt
padding(sentence::Vector{String}, bptt::Integer=70) = (length(sentence) <= bptt) ? cat(sentence, repeat(["<pos>"], bptt-length(sentence)); dims = 1) : sentence[1:bptt]

function generator(c::Channel, sentences = sentences; batchsize::Integer=70, bptt::Integer=70)
    num_sents = Int(floor(length(sentences)/batchsize))
    for i=1:num_sents
        batch = tokenize.([sentences[i+Int(num_sents*j)] for j=0:batchsize-1])
        batch = intern.(padding.(batch, bptt+1))
        X, Y = broadcast(x -> getindex(x, 1:bptt), batch), broadcast(x -> getindex(x, 2:bptt+1), batch)
        put!(c, (X, Y))
    end
end

function loss(x, y)
    h = softmax(x)
    loss = crossentropy(h, y)
    Flux.reset!(lm)
    return loss
end

Flux.train!((x, y) -> loss(x, y, lm))

function train!(lm::LanguageModel; batchsize::Integer=70, bptt::Integer=70,
    gradient_clip::Float64=0.25, initLearnRate::Number=30; epochs::Integer=1)

    sentences = preprocess()
    gen = Channel(x -> generator(x, sentences; batchsize = batchsize, bptt = bptt))
    for epoch=1:epochs
        X, Y = take!(gen)
        X, Y = broadcast(x -> onehot(x, lm.vocab), X), broadcast(y -> hcat(onehot(y, lm.vocab)...), Y)

        emDropMat = embeddingDropout(lm)
        X = broadcast(x -> embeddings(x, emDropMat, lm), X)    #vector of matrices of embeddings of input sentences
        X = lm.wordDropout.(X)
        X = lm.RecurrentLayers.(X) #outputMat is a vector of matrices of output sentences
        X = TiedEmbeddings(X, emDropMat)

        grads = broadcast((x, y) -> Tracker.gradient(() -> loss(x, y), params(lm)), X, Y)
        for i=1:batchsize
            update!(lm.embedMat, -lr .* grads[lm.embedMat])
            for cell in [lm.lstmLayer1.cell, lm.lstmlayer2.cell, lm.lstmLayer3.cell]
                update!(cell.Wi, -lr .* grads[cell.Wi])
                update!(cell.Wh, -lr .* grads[cell.Wh])
                update!(cell.b, -lr .* grads[cell.b])
                update!(cell.h, -lr .* grads[cell.h])
                update!(cell.c, -lr .* grads[cell.c])
            end
        end
    end
end
