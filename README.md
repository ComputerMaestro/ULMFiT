# ULMFiT

This is the Julia implementation of [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf) paper released by the Jeremy Howard and Sebastian Ruder.

Checkout these [blogs](https://nextjournal.com/ComputerMaestro) for better understanding of the implementation.

## Data Loading and Preprocessing

These steps are necessary to starting training ULMFiT for any task. For pretraining step for Language model, a general-purpose corpus is needed, which here is WikiText-103 by default. Similarly, for fine-tuning Language Model and fine-tuning classifier we need a dataset for the specific task (example Sentiment Analysis, Topic classification etc). For all of these steps, the data is to be loaded for training and for that data loaders are to be defined. Since the data used to train for such a large model is large, so it is not recommended to load all the data at once, instead the data should be loaded in batches through tasks concept of julia (Refer [this](https://docs.julialang.org/en/v1.0/manual/control-flow/#man-tasks-1) documentation) using `Channels`. Basically, we need to create `Channels` which supply a mini-batch at every call.

Before dividing the tokenized corpus into batches, first preprocess the corpus properly according to the vocabulary used for the model. Try to reduce the number of UNKNOWN tokens as much as possible by splitting tokens like "necessity." to "necessity" and "." , "name"" to "name" and """, "hall's" to "hall" and "'s'" etc. it is necessary to look into the tokenized corpus to reduce such words, otherwise, excess of such tokens will lead to increase in the count of UNKNOWN tokens which eventually would result in improper learning of model. Also, it is essential to use lower casing if there is no special token defined in vocabulary of model to specify upper cased words to the model.
Apart from this, apply standard techniques for text preprocessing depending on your requirements. Also, to ensure whatever the model is learning is correct for checking on validation set or else Sampling should be done after every epoch or a after a descent number of iterations. A `sampling` function is provided for the same:

```julia
julia> sampling(starting_text::String, lm::LanguageModel)
```

Arguments:

 * `starting_text` : This is the initial piece of text given to the Language Model after which the Language Model will generate text in continuation
 * `lm`            : Instance of `LanguageModel struct`

Mini-batch format for pretraining step:

Firstly, get cleared about what are `batchsize` and `bptt` from this [blog](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-2) then a generator function is to created which is basically a `Channel` which will output the `X` and `Y` for a mini-batch at once.
Now, for the pretraining language model and fine-tuning of language model both steps are semi-supervised learning steps so the `Y` here is nothing but the next succeeding word for every word in any sequence in `X`. For example:

```julia
# a sequence 'x'
julia> x = ["this", "is", "an", "example", "sequence"]

# Corresponding 'y' sequence
julia> y = ["is", "an", "example", "sequence", "."]
```

This is just an example of a simple `x` and  `y` set, in practical `X` will be a `Vector` of `Vector`s (a mini-batch) where each `Vector` length will be equal to batchsize and it will contain the words for one timestep of all sequences in that mini-batch and it's succeeding vector will contain the words for next timestep of all the corresponding sequences. Similarly, `Y` will also be a `Vector` of `Vector`s but it will start from the second `Vector` in `X` and end at the next succeeding `Vector` of words for the last timestep in the `X`. For example:

```julia
julia> gen = load_wikitext_103(4, 10)     # Loads WikiText-103 and outputs a Channel to give a mini-batch(of given batchsize and bptt) at each call
Channel{Any}(sz_max:0,sz_curr:1)

julia> num_of_batches = take!(gen);      # here the first thing that generator gives is number of batches which it can output

julia> X = take!(gen)
10-element Array{Array{Any,1},1}:
 ["senjō", ",", "indicated", "after"]   
 ["no", "he", ",", "two"]               
 ["valkyria", "sent", "\"", "games"]    
 ["3", "a", "i", ","]                   
 [":", "formal", "am", "making"]        
 ["<unk>", "demand", "to", "a"]         
 ["chronicles", "for", "some", "start"]
 ["(", "surrender", "extent", "against"]
 ["japanese", "of", "influenced", "the"]
 [":", "the", "by", "vancouver"]

 julia> Y = take!(gen)
 10-element Array{Array{Any,1},1}:
 ["no", "he", ",", "two"]                    
 ["valkyria", "sent", "\"", "games"]         
 ["3", "a", "i", ","]                        
 [":", "formal", "am", "making"]             
 ["<unk>", "demand", "to", "a"]              
 ["chronicles", "for", "some", "start"]      
 ["(", "surrender", "extent", "against"]     
 ["japanese", "of", "influenced", "the"]     
 [":", "the", "by", "vancouver"]             
 ["戦場のヴァルキュリア", "arsenal", "them", "canucks"]
```

For Language Model Fine-tuning step, the above formats for `X` and `Y` will be followed as such, just the corpus or examples used for the fine-tuning will change.

For Fine-tuning of Classifier, the format of `X` will not change but now since this is not Language Modelling task anymore so `Y` will not be the `Vector` of words anymore, rather it will be a `Vector` of target classes for each sequence in that mini-batch (`X`).

## Training ULMFiT
### Step 1 - Pre-training Language Model

Checkout [blog1](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-for-text-clasification) and [blog2](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-2) for the conceptual understanding of this step.
The repo contains `pretrain_lm.jl` file pre-training a Language model from scratch based on given corpus. Here, the corpus pre-processed in the above step is used here to train the language model. (Refer this [blog](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-2) for batchsize and bptt understanding).

#### To start training run:

```julia
julia> fit!(lm::LanguageModel,
            model_layers=nothing;
            batchsize::Integer=64,
            bptt::Integer=70,
            gradient_clip::Float64=0.25,
            base_lr=0.004,
            epochs::Integer=1,
            checkpoint_itvl::Integer=5000)
```

Positional Arguments:

 * `lm`               : instance of LanguageModel
 * `model_layers`     : `Chain` of embedding, RNN and Dropout layers for training

Keyword Arguments:

 * `batchsize`        : number examples pass from the model at a time while training
 * `bptt`             : length of sequences passing at one pass (in one mini-batch)
 * `gradient_clip`    : upper bound for all the gradient values
 * `base_lr`          : base learning rate or the maximum value for the slanted triangular learning rates
 * `epochs`           : number of epochs
 * `checkpoint_itvl`  : Stands for Checkpoint interval, interval of number of iterations after which the model weights are saved to a specified BSON file

### Step 2 - Fine-tuning Language Model

Checkout this [blog](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-3) for better conceptual understanding before starting the process.
In repo, the `fineTuneLM.jl` file is used for the fine-tuning step. This step will fine-tune the pre-trained language model for downstream task. Dataset used here will be preprocessed as specified above.

#### To start fine-tuning the Language model by running:

```julia
julia> fineTuneLM(lm::LanguageModel;
        batchsize::Integer=64,
        bptt::Integer=70,
        gradient_clip::Float64=0.25,
        stlr_cut_frac::Float64=0.1,
        stlr_ratio::Float32=32,
        stlr_η_max::Float64=0.01,
        epochs::Integer=1,
        checkpoint_itvl::Integer=5000)
```

Positional Arguments:

 * `lm`               : Instance of `LanguageModel struct`

Keyword Arguments:

 * `batchsize`        : number examples pass from the model at a time while fine-tuning
 * `bptt`             : length of sequences passing at one pass (in one mini-batch)
 * `gradient_clip`    : upper bound for all the gradient values
 * `stlr_cut_frac`    : In STLR, it is the fraction of iterations for which LR is increased
 * `stlr_ratio`       : In STLR, it specifies how much smaller is lowest LR from maximum LR
 * `stlr_η_max`       : In STLR, this is the maximum LR value
 * `epochs`           : It is simply the number of epochs for which the language model is to be fine-tuned
 * `checkpoint_itvl`  : Stands for Checkpoint interval, interval of number of iterations after which the model weights are saved to a specified BSON file

For Sentiment analysis with IMDB dataset, `imdb_fine_tune_data` funciton is provided to load data for Fine-tuning Language Model:

```julia
# This outputs a generator same as used for pre-training of language model
# The data given by this generator is from the `unsup` part of the IMDB dataset (refer README of the IMDB)
julia> ft_gen = imdb_fine_tune_data(4, 10)
Channel{Any}(sz_max:0,sz_curr:1)
```

Arguments:

 * `batchsize`    : Number of sequences passing at a time
 * `bptt`         : Number of tokens in each sequence of mini-batch
 * `num_examples` : (optional) Number of examples to be taken from all unsup examples of IMDB dataset

### Step 3 - Fine-tuning the classifier for downstream task

Checkout this [blog](https://nextjournal.com/ComputerMaestro/jsoc19-practical-implementation-of-ulmfit-in-julia-4) for better conceptual understanding of this step.
In the repo, the `TextClassifier.jl` file is used for the fine-tuning classifier for any number of classes after the above fine-tuning step is over.

#### To start fine-tuning the classifier run:

```julia
julia> train_classifier(lm::LanguageModel,
        batchsize::Integer=64,
        bptt::Integer=70,
        gradient_clip::Float64=0.25,
        stlr_cut_frac::Float64=0.1,
        stlr_ratio::Number=32,
        stlr_η_max::Float64=0.01,
        epochs::Integer=1,
        checkpoint_itvl::Integer=5000)
```

Positional Arguments:

* `lm`               : Instance of `LanguageModel struct`
* `classes`          : Size of output layer for classifier or number of classes for which the classifier is to be trained
* `hidden_layer_size`: Size of the hidden linear layer added for making classifier

Keyword Arguments:

* `batchsize`        : number examples pass from the model at a time while fine-tuning
* `bptt`             : length of sequences passing at one pass (in one mini-batch)
* `gradient_clip`    : upper bound for all the gradient values
* `stlr_cut_frac`    : In STLR, it is the fraction of iterations for which LR is increased
* `stlr_ratio`       : In STLR, it specifies how much smaller is lowest LR from maximum LR
* `stlr_η_max`       : In STLR, this is the maximum LR value
* `epochs`           : It is simply the number of epochs for which the language model is to be fine-tuned
* `checkpoint_itvl`  : Stands for Checkpoint interval, interval of number of iterations after which the model weights are saved to a specified BSON file

By Default the Text Classifier can be fine-tuned for the sentiment analysis task.
To train it for the sentiment analysis with IMDB dataset, `imdb_classifier_data` function is provided to load the training examples of the IMDB dataset:

```julia
# This funciton outputs a Channel, which outputs ont example at a time
# Example can be negative or positive randomly
julia> classifier_data_gen = imdb_classifier_data()
```
