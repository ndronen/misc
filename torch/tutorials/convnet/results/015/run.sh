#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th -i doall.lua \
    -type cuda \
    -loss mse \
    -scaleMseTarget 25 \
    -save $RESULTS \
    -size all \
    -nKernels 500 \
    -kernelWidth 4 \
    -nFullyConnectedLayers 125,75,25,10 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -weightDecay 0.0001 \
    -batchSize 128 \
    -maxNorm 2 \
    -wordDims 20 \
    -test
