#!/bin/bash -ex

# Example of calling bin/token-activations.lua.

bin/token-activations.lua results/test-wiki/okanohara/1.9m/model.net \
    /tmp/sents-okanohara-wiki-dlm-train-initial-index.json \
    results/test-wiki/okanohara/1.9m/filter-info-test-data.h5 \
    'What is a practical skill?'
