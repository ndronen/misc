#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

# Fewer kernels, 2-grams, 3 fully-connected layers, 
# no momentum, no weight decay.

th doall.lua \
    -type cuda \
    -loss nll \
    -save $RESULTS \
    -size all \
    -nKernels 200 \
    -kernelWidth 2 \
    -nFullyConnectedLayers 3 \
    -learningRate 0.1 \
    -batchSize 128 \
    -maxNorm 2
