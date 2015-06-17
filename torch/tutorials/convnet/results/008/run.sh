#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th doall.lua \
    -type cuda \
    -loss nll \
    -save $RESULTS \
    -size all \
    -nKernels 500 \
    -kernelWidth 4 \
    -nFullyConnectedLayers 1000,900,1000 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -batchSize 128 \
    -weightDecay 0.001 \
    -maxNorm 2
