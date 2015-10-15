#!/bin/bash -xe

N=60000000

./train-model.py \
    models/preposition/convnet \
    data/prepositions-all-new-train-$N/prepositions-all-new-train-$N-00.h5 \
    data/prepositions-all-new-validate.h5 \
    Xwindow \
    --target-name original_word_code \
    --target-data data/prepositions-all-new-target-data.json \
    --description "input = Xwindow, target = original_word_code, random word vectors, contrasting, $N training examples, AdaGrad, dropout, batch normalization, n_filters=3000, n_hidden=2000, n_word_dims=50, 4 fully-connected layers" \
    --n-vocab 100000 \
    --n-epochs 10 \
    --model-cfg optimizer=Adagrad regularization_layer="dropout+normalization" patience=120 n_filters=3000 n_hidden=2000 n_word_dims=50 \
    --n-validation 20000 \
    --classification-report \
    --extra-train-file $(ls data/prepositions-all-new-train-$N/* | grep -v 00.h5) \
    --log
