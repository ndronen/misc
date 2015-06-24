#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th bin/run.lua 150001 /tmp/sents-okanohara-wiki-dlm-train-initial-1867900-idx.h5 \
    -save $RESULTS \
    -type cuda \
    -nKernels 250 \
    -kernelWidth 4 \
    -nFullyConnectedLayers 500,200 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -batchSize 128 \
    -test \
    -testFile /tmp/sents-okanohara-wiki-dlm-test-initial-40000-idx.h5 \
    -nTrain 500000 \
    -lookupTableDropout 0.0
