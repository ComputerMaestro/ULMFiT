"""
ULMFiT - testing file
"""

# Test the language model on the validation set
function test_lm(lm, data_gen)
    model_layers = Chain(
        lm.embedding_layer,
        lm.lstm_layer1,
        lm.lstm_layer2,
        lm.lstm_layer3,
        x -> lm.embedding_layer(x, true),
        softmax
    )
    testmode!(model_layers)
    sum_l, l_vect = 0, []
    for iter=1:num_of_iters
        x, y = take!(data_gen)
        h = broadcast(w -> model_layers[1](indices([w], lm.vocab, "_unk_")), x)
        h = model_layers[2:end].(h)
        y = broadcast(x -> Flux.onehotbatch([x], lm.vocab, "_unk_"), y)
        Flux.reset!(model_layers)
        l = sum(crossentropy.(Tracker.data.(h), y))
        sum_l += l
        push!(l_vect, l)
    end
    return sum_l/num_of_iters, l_vect
end

# Sampling
function sampling(starting_text::String, lm::LanguageModel=LanguageModel())
    model_layers = Chain(
        lm.lstm_layer1,
        lm.lstm_layer2,
        lm.lstm_layer3,
    )
    testmode!(model_layers)
    tokens = tokenize(starting_text)
    word_indices = map(x -> indices([x], lm.vocab, "_unk_"), tokens)
    embeddings = lm.embedding_layer.(word_indices)
    h = (model_layers.(embeddings))[end]
    probabilities = softmax(lm.embedding_layer(h, true))
    prediction = lm.vocab[findall(isequal(maximum(probabilities)), probabilities)[1]]
    println("SAMPLING...")
    print(prediction, ' ')
    while true
        h = lm.embedding_layer(indices([prediction], lm.vocab, "_unk_"))
        h = model_layers(h)
        probabilities = softmax(lm.embedding_layer(h, true))
        prediction = lm.vocab[findall(isequal(maximum(probabilities)), probabilities)[1]]
        print(prediction, ' ')
        prediction == "_pad_" && break
    end
end

# Test Classifier
function test_classifier(classifier::TextClassifier, data_gen)
    testmode!(classifier)
    sum_l, l_vect = 0, []
    for iter=1:num_of_iters
        x = take!(data_gen)
        y = take!(data_gen)
        h = broadcast(x -> classifier.rnn_layers[1](indices([x], lm.vocab, "_unk_")), x)
        h = classifier.rnn_layers[2:end].(h)
        h = classifier.linear_layers(h)
        Flux.reset!(classifier)
        l = crossentropy(Tracker.data(h), y)
        sum_l += l
        push!(l_vect, l)
    end
    return sum_l/num_of_iters, l_vect
end
