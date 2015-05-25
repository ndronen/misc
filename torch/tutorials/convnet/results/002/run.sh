#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

# This experiment differs from 001 in that it uses a max norm of 10.
# Based on the results of 001, this should first affect the weights
# of the output layer, which grew large while the weights of the other
# layers mostly shrunk. 

th doall.lua \
    -type cuda \
    -loss nll \
    -save $RESULTS \
    -size all \
    -nKernels 1000 \
    -kernelWidth 3 \
    -nFullyConnectedLayers 2 \
    -weightDecay 0.001 \
    -momentum 0.9 \
    -learningRate 0.001 \
    -batchSize 32 \
    -maxNorm 10
