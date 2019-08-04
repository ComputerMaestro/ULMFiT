"""
Helping functions
"""

# Used for data loadng
using Random

# Converts vector of words to vector of indices
function indices(wordVect::Vector, vocab::Vector, unk)
    function index(x, unk)
        idx = something(findfirst(isequal(x), vocab), 0)
        idx > 0 || return findfirst(isequal(unk), vocab)
        return idx
    end
    return broadcast(x -> index(x, unk), wordVect)
end

#Adding "_pad_" keyowrd at the end if the length of the sentence is < bptt
function padding(batches::Vector)
    n = maximum([length(x) for x in batches])
    return ([length(batch) < n ? cat(batch, repeat(["_pad_"], n-length(batch)); dims = 1) : batch[1:n] for batch in batches], n)
end

# To initialize funciton for model LSTM weights
init_weights(extreme::AbstractFloat, dims...) = randn(Float32, dims...) .* sqrt(Float32(extreme))

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

# custom mean function
mean(nums...) = sum(nums)/length(nums)

"""
Default Data Loaders for ULMFiT training for Sentiment Analysis
 - WikiText-103 corpus is to pre-train the Language model
 - IMDB movie review dataset - unsup data is used for fine-tuning Language Mode for Sentiment Analysis
 - IMDB movie review dataset - labelled data is used for training classifier for Sentiment Analysis
"""
# WikiText-103 corpus loader
function loadCorpus()
    corpuspath = joinpath(datadep"WikiText-103", "wiki.valid.tokens")
    corpus = read(open(corpuspath, "r"), String)
    return intern.(tokenize(corpus))
end

# IMDB Data loaders for Sentiment Analysis
# IMDB data loader for fine-tuning Language Model
function imdb_fine_tune_data()
    targetData = read(open(filepath, "r"), String)
    return intern.(tokenize(targetData))
end

# IMDB data loader for training classifier
function imdb_classifier_data()
    basepath = joinpath(datadep"IMDB movie review dataset", "aclImdb/train")
    filepaths = readdir(joinpath(basepath, "neg"))
    append!(filepaths, readdir(joinpath(basepath, "pos")))
    shuffle!(filepaths)
    Channel(csize=4) do docs
        put!(docs, length(filepaths))
        for path in filepaths   #extract data from the files in directory and put into channel
            open(path) do fileio
                cur_text = read(fileio, String)
                sents = [intern.(tokenize(sent)) for sent in split_sentences(cur_text)]
                put!(docs, sents)
            end #open
        end #for
    end #channel
end
