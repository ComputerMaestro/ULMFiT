"""
ULMFiT - Text Classifier
"""

include("custom_layers.jl")     # including custome layers created for ULMFiT model
include("utils.jl")     # including utilities
include("fineTuneLM.jl")        # including useful functions from fine-tuning file

# Get classfier layers in a chain
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
    Flux.reset!(model_layers)
    return l
end

# Training of classifier
function train_classifier(lm::LanguageModel, batchsize::Integer=64, bptt::Integer=70)
    classifier = get_classifier(lm)
    trainable = classifier[[1, 3, 5, 7, 9, 12]]
    opts = [ADAM(0.001, (0.7, 0.99)) for i=1:length(trainable)]
    gpu!.(classifier)

    for epoch=1:epochs
        gen = imdb_classifier_data()
        num_of_iters = take!(gen)
        T = Int(floor((num_of_iters*2)/100))
        set_trigger!.(T, [lm.lstm_layer1, lm.lstm_layer2, lm.lstm_layer3])
        for iter=1:num_of_iters
            l = loss(lm, classifier, gen, α, β)

            # Slanted triangular learning rates
            cut = T * stlr_cut_frac
            p_frac = (iter < cut) ? iter/cut : (1 - ((iter-cut)/(cut*(1/stlr_cut_frac-1))))
            ηL = stlr_η_max*((1+p_frac*(stlr_ratio-1))/stlr_ratio)

            # Gradual-unfreezing Step with discriminative fine-tuning
            epoch < length(trainable) ? discriminative_step!(trainable[end-epoch+1:end], ηL, l, gradient_clip, opts[end-epoch+1:end]) : discriminative_step!(trainable, ηL, l, gradient_clip, opts)

            # ASGD Step
            asgd_step!.(iter, [lm.lstm_layer1, lm.lstm_layer2, lm.lstm_layer3])

            println("loss:", l, "epoch: ", epoch, "iteration: ", iter)
        end
    end
end
