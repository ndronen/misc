#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th -i doall.lua \
    -type cuda \
    -loss mse \
    -scaleMseTarget 25 \
    -save $RESULTS \
    -size all \
    -nKernels 1000 \
    -kernelWidth 2 \
    -nFullyConnectedLayers 250,125,50 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -weightDecay 0.0002 \
    -batchSize 128 \
    -maxNorm 2 \
    -wordDims 25 \
    -test

