#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th -i doall.lua \
    -type cuda \
    -loss nll \
    -save $RESULTS \
    -size all \
    -nKernels 1000 \
    -kernelWidth 2 \
    -nFullyConnectedLayers 750,750,750,750 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -batchSize 128 \
    -wordDims 25 \
    -maxNorm 2 \
    -maxWordNorm 1 \
    -activation tanh \
    -test
