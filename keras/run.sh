#!/bin/bash -xe

N=10000000

./train-model.py \
    models/preposition/convnet \
    data/prepositions-all-new-train-$N.h5 \
    data/prepositions-all-new-validate.h5 \
    Xwindow \
    --target-name original_word_code \
    --target-data data/prepositions-all-new-target-data.json \
    --description "input = Xwindow, target = original_word_code, random word vectors, contrasting, $N training examples, AdaGrad, dropout, batch normalization, n_filters=100, n_hidden=200, n_word_dims=10, 2 fully-connected layers, 10 epochs" \
    --n-vocab 100000 \
    --n-epochs 10 \
    --model-cfg optimizer=Adagrad regularization_layer="dropout+normalization" patience=10 n_filters=100 n_hidden=200 n_word_dims=10 \
    --n-validation 20000 \
    --log

./train-model.py \
    models/preposition/convnet \
    data/prepositions-all-new-train-$N.h5 \
    data/prepositions-all-new-validate.h5 \
    Xwindow X \
    --target-name original_word_code \
    --target-data data/prepositions-all-new-target-data.json \
    --description "input = Xwindow X, target = original_word_code, random word vectors, contrasting, $N training examples, AdaGrad, dropout, batch normalization, n_filters=100, n_hidden=200, n_word_dims=10, 2 fully-connected layers, 10 epochs" \
    --n-vocab 100000 \
    --n-epochs 10 \
    --model-cfg optimizer=Adagrad regularization_layer="dropout+normalization" patience=10 n_filters=100 n_hidden=200 n_word_dims=10 \
    --n-validation 20000 \
    --log

#./train-model.py \
#    models/preposition/convnet \
#    data/prepositions-all-new-train-$N.h5 \
#    data/prepositions-all-new-validate.h5 \
#    X \
#    --target-name original_word_code \
#    --target-data data/prepositions-all-new-target-data.json \
#    --description "input = Xwindow, target = original_word_code, random word vectors, contrasting, $N training examples, AdaGrad, dropout, batch normalization, n_filters=100, n_hidden=200, n_word_dims=10, 2 fully-connected layers, 10 epochs" \
#    --n-vocab 100000 \
#    --n-epochs 10 \
#    --model-cfg optimizer=Adagrad regularization_layer="dropout+normalization" patience=10 n_filters=100 n_hidden=200 n_word_dims=10 \
#    --n-validation 20000 \
#    --log


#./train-model.py \
#    models/preposition/convnet \
#    data/prepositions-all-new-train-$N.h5 \
#    data/prepositions-all-new-validate.h5 \
#    X \
#    --target-name original_word_code \
#    --target-data data/prepositions-all-new-target-data.json \
#    --description "input = X (whole sentence only), target = original_word_code, random word vectors, contrasting, $N training examples, Adagrad, dropout, batch normalization, n_filters=3000, filter_width=3, h_hidden=500, shuffled training data, 4 fully-connected layers" \
#    --n-vocab 100000 \
#    --model-cfg optimizer=Adagrad regularization_layer="dropout+normalization" patience=120 n_filters=3000 filter_width=3 n_hidden=500 \
#    --shuffle \
#    --n-validation 20000 \
#    --classification-report \
#    --extra-train-file $(ls data/prepositions-all-new-train-$N/* | grep -v 00.h5) \
#    --log
