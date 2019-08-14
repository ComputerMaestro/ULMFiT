"""
Helping functions
"""

# Converts vector of words to vector of indices
function indices(wordVect::Vector, vocab::Vector, unk)
    function index(x, unk)
        idx = something(findfirst(isequal(x), vocab), 0)
        idx > 0 || return findfirst(isequal(unk), vocab)
        return idx
    end
    return broadcast(x -> index(x, unk), wordVect)
end

#Padding multiple sequences w r t the max size sequence
function pre_pad_sequences(sequences::Vector, pad::String="_pad_")
    max_len = maximum([length(x) for x in sequences])
    return [[fill(pad, max_len-length(sequence)); sequence] for sequence in sequences]
end

function post_pad_sequences(sequences::Vector, pad::String="_pad_")
    max_len = maximum([length(x) for x in sequences])
    return [[sequence; fill(pad, max_len-length(sequence))] for sequence in sequences]
end

# To initialize funciton for model LSTM weights
init_weights(extreme::AbstractFloat, dims...) = randn(Float32, dims...) .* sqrt(Float32(extreme))

# Generator, whenever it is called it gives one mini-batch
function generator(c::Channel, corpus; batchsize::Integer=64, bptt::Integer=70)
    X_total = post_pad_sequences(chunk(corpus, batchsize))
    n_batches = Int(floor(length(X_total[1])/bptt))
    put!(c, n_batches)
    for i=1:n_batches
        start = bptt*(i-1) + 1
        batch = [Flux.batch(X_total[k][j] for k=1:batchsize) for j=start:start+bptt]
        put!(c, batch[1:end-1])
        put!(c, batch[2:end])
    end
end

# Sequence Bucketing
# function sequence_bucketing(buckets::Integer, lengths::Vector{Integer}, k::Integer)
#     f = [count(x -> x == l, lengths) for l=1:maximum(lengths)]
#     n = length(f)
#     prev_dp = fill(fill(0, n-1), buckets)
#     dp = fill(convert(Array{Union{Int64, Float64}, 1}, fill(0, n)), buckets)
#     for i=1:buckets
#         dp[i][0] = Inf
#         (i-2 > 0) && (dp[i][k] = minimum([(dp[j][k-1] + i*(sum([f[t] for t=i+1:i]))) for j=1:i-2]))
#     end
#     for q=1:buckets
#         for i=1:n
#             cur_sum = f[i]
#             for j=i-1:1
#                 val = cur_sum*i + dp[q-1][j]
#                 if val < dp[q][i]
#                     dp[q][i] = val
#                     prev_dp[q][i] = j
#                 end
#                 cur_sum = cur_sum + f[j]####prob
#             end
#         end
#     end
#     cur_id = n-1
#     bests = []
#     for i=buckets:1
#         bests = append!([cur_id], bests)
#         cur_id = prev_dp[i][cur_id]
#     end
#     return bests
# end
