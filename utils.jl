"""
Helping functions
"""

gpu!(entity) = nothing
cpu!(entity) = nothing

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

# custom mean function
mean(nums...) = sum(nums)/length(nums)
