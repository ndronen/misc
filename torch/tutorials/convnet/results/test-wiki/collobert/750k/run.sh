#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th bin/run.lua 150001 data/sents-okanohara-wiki-dlm-not-used-train-1000000-idx.h5 \
    -save $RESULTS \
    -type cuda \
    -nKernels 250 \
    -kernelWidth 4 \
    -nFullyConnectedLayers 500,200 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -batchSize 128 \
    -test \
    -testFile data/sents-okanohara-wiki-dlm-not-used-test-0020000-idx.h5 \
    -nTrain 750000 \
    -lookupTableDropout 0.0 \
    -makeNegativeExamples
