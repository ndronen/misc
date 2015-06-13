#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th doall.lua \
    -type cuda \
    -loss nll \
    -save $RESULTS \
    -size all \
    -nKernels 2000 \
    -kernelWidth 3 \
    -nFullyConnectedLayers 1 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -batchSize 128 \
    -maxNorm 2
