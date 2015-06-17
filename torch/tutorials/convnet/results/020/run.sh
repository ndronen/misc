#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th -i doall.lua \
    -type cuda \
    -loss nll \
    -save $RESULTS \
    -size all \
    -nKernels 2000 \
    -kernelWidth 3 \
    -nFullyConnectedLayers 500 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -batchSize 128 \
    -wordDims 25 \
    -maxNorm 1 \
    -maxWordNorm 1 \
    -activation relu \
    -test \
    -renormFreq 4
