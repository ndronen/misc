#!/bin/bash -xe

RESULTS="$(dirname -- $(readlink -f -- "$0"))"

# This experiment differs from 002 in that it uses a max norm of 1. 
# Based on the results of 002, this should first affect the weights
# of the output layer, which grew large while the weights of the other
# layers mostly shrunk. 

# Also increased the batch size from 32 to 128, based on recommendations
# here: http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html

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
    -learningRate 0.1 \
    -batchSize 128 \
    -maxNorm 1
