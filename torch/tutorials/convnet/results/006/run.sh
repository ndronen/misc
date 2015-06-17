#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

th doall.lua \
    -type cuda \
    -loss nll \
    -save $RESULTS \
    -size all \
    -nKernels 200 \
    -kernelWidth 3 \
    -nFullyConnectedLayers 500,400,600 \
    -learningRate 0.1 \
    -momentum 0.9 \
    -batchSize 128 \
    -maxNorm 2
