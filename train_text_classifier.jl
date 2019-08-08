"""
ULMFiT - Text Classifier
"""

cd(@__DIR__)
include("custom_layers.jl")     # including custome layers created for ULMFiT model
include("utils.jl")     # including utilities
include("fine_tune_lm.jl")        # including useful functions from fine-tuning file

mutable struct TextClassifier
    rnn_layers::Flux.Chain
    linear_layers::Flux.Chain
end

function TextClassifier(lm::LanguageModel=load_model!(), clsfr_out_sz::Integer=1, clsfr_hidden_sz::Integer=50, clsfr_hidden_drop::Float64=0.0)
    return TextClassifier(Chain(
            lm.embedding_layer,
            VarDrop(lm.wordDropProb),
            lm.lstm_layer1,
            VarDrop(lm.LayerDropProb),
            lm.lstm_layer2,
            VarDrop(lm.LayerDropProb),
            lm.lstm_layer3,
            VarDrop(lm.FinalDropProb)),
        Chain(
            gpu(PooledDense(length(lm.lstm_layer3.layer.cell.h), clsfr_hidden_sz, relu)),
            gpu(BatchNorm(clsfr_hidden_sz, relu)),
            Dropout(clsfr_hidden_drop),
            gpu(Dense(clsfr_hidden_sz, clsfr_out_sz)),
            softmax
        )
    )
end

# Forward step for classifier
function forward(lm, classifier, gen)
    batch = broadcast(x -> gpu(classifier.rnn_layers[1](indices(x, lm.vocab, "_unk_"))), batch)
    batch = classifier.rnn_layers[2:end].(batch)
    batch = classifier.linear_layers(batch)
    return batch
end

# Loss function for classifier
function loss(lm, classifier, gen)
    H = forward(lm, classifier.rnn_layers, take!(gen))
    Y = gpu.(take!(gen))
    l = sum(crossentropy.(H, Y))
    Flux.reset!(model_layers)
    return l
end

# Training of classifier
function train_classifier!(lm::LanguageModel, classifier::TextClassifier=TextClassifier(lm), classes::Integer=2, hidden_layer_size::Integer=50; batchsize::Integer=64, bptt::Integer=70, gradient_clip::Float64=0.25, stlr_cut_frac::Float64=0.1, stlr_ratio::Number=32, stlr_η_max::Float64=0.01, epochs::Integer=1, checkpoint_itvl=5000)
    trainable = []
    append!(trainable, [classifier.rnn_layers[[1, 3, 5, 7]]...])
    append!(trainable, [classifier.linear_layers[[1, 4]]...])
    opts = [ADAM(0.001, (0.7, 0.99)) for i=1:length(trainable)]
    gpu!.(classifier.rnn_layers)

    for epoch=1:epochs
        gen = imdb_classifier_data()
        num_of_iters = take!(gen)
        T = num_of_iters-Int(floor((num_of_iters*2)/100))
        set_trigger!.(T, [lm.lstm_layer1, lm.lstm_layer2, lm.lstm_layer3])
        cut = num_of_iters * stlr_cut_frac
        for iter=1:num_of_iters
            l = loss(lm, classifier, gen)

            # Slanted triangular learning rates
            t = iter + (epoch-1)*num_of_iters
            p_frac = (iter < cut) ? iter/cut : (1 - ((iter-cut)/(cut*(1/stlr_cut_frac-1))))
            ηL = stlr_η_max*((1+p_frac*(stlr_ratio-1))/stlr_ratio)

            # Gradual-unfreezing Step with discriminative fine-tuning
            epoch < length(trainable) ? discriminative_step!(trainable[end-epoch+1:end], ηL, l, gradient_clip, opts[end-epoch+1:end]) : discriminative_step!(trainable, ηL, l, gradient_clip, opts)

            # ASGD Step
            asgd_step!.(iter, [lm.lstm_layer1, lm.lstm_layer2, lm.lstm_layer3])

            reset_masks!.(classifier.rnn_layers)

            println("loss:", l, "epoch: ", epoch, "iteration: ", iter)
        end
    end
end
