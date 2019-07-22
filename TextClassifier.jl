"""
ULMFiT - Text Classifier
"""

include("awd_lstm.jl")
include("utils.jl")
include("fineTuneLM.jl")

function get_classifier(lm::LanguageModel=load_model!(), clsfr_out_sz::Integer=2, clsfr_hidden_sz::Integer=50)
    return Chain(
        lm.embedding_layer,
        VarDrop(lm.wordDropProb),
        lm.lstm_layer1,
        VarDrop(lm.LayerDropProb),
        lm.lstm_layer2,
        VarDrop(lm.LayerDropProb),
        lm.lstm_layer3,
        VarDrop(lm.FinalDropProb),
        PooledDense(length(lm.lstm_layer3.layer.cell.h), clsfr_hidden_sz, relu),
        BatchNorm(clsfr_hidden_sz, relu),
        Dropout(clsfr_hidden_drop),
        Dense(clsfr_hidden_sz, clsfr_out_sz),
        softmax
    )
end

function gradual_freezing(unfreezed_layers, l, opt)
    l = Tracker.hook(x -> grad_clipping(x, gradient_clip), l)
    grads = Tracker.gradient(() -> l, params(unfreezed_layers))
    Tracker.update!(opt, params(unfreezed_layers), grads)
end

# Forward step for classifier
function forward(lm, encoder, decoder, gen, α, β)
    batch = broadcast(x -> gpu(encoder[1](indices(x, lm.vocab, "_unk_"))), batch)
    batch = encoder[2:7].(batch)
    tar_value = calc_tar(batch, β)
    batch = encoder[8].(batch)
    ar_value = calc_ar(batch, α)
    batch = decoder(batch)
    return batch, ar_value, tar_value
end

# Loss function for classifier
function loss(lm, classifier, gen, α, β)
    H, ar_value, tar_value = forward(lm, classifier[1:8], classifier[9:end], take!(gen), α, β)
    Y = gpu.(take!(gen))
    l = sum(crossentropy.(H, Y)) + ar_value + tar_value
    Flux.truncate!(model_layers)
    return l
end

# Training of classifier
function train_classifier(lm::LanguageModel, batchsize::Integer=64, bptt::Integer=70)
    classifier = get_classifier(lm)
    trainable = classifier[[1, 3, 5, 7, 9, 12]]
    opt = ADAM(0.01, (0.7, 0.99))
    gpu!.(classifier)

    for epoch=1:epochs
        for iter=1:num_of_iters
            l = loss(lm, classifier, gen, α, β)
            epoch < length(trainable) ? gradual_freezing(trainable[end-epoch+1:end], l, opt) : gradual_freezing(trainable, l, opt)
        end
    end
end
