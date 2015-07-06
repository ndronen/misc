#!/bin/bash -e

# Example of calling bin/token-activations.lua.

# The path to a serialized torch model.
#model_file=results/test-wiki/okanohara/1.9m/model.net
model_file=results/test-wiki/collobert/1m/model.net

# The path to a JSON file containing vocabulary used for mapping tokens
# to indices and vice-versa.
#index_file=/tmp/sents-okanohara-wiki-dlm-train-initial-index.json
index_file=data/sents-okanohara-wiki-dlm-not-used-index.json

# The path to an HDF5 file containing information about the model's
# filters, such as whether they have learned to recognize features
# positive examples or negative examples.  You can create this file 
# for a particular model by running bin/classify-filters.lua.
#data_file=results/test-wiki/okanohara/1.9m/filter-info-test-data.h5
data_file=results/test-wiki/collobert/1m/filter-info-test-data.h5

# A sentence to be processed by the model.
sentence="Four score and seven years ago "\
"our fathers brought forth on this continent, "\
"a new nation, conceived in Liberty, "\
"and dedicated to the proposition that all men are created equal."

#sentence="The greatest problem facing any organism is successful reaction to its environment."
#sentence="Cheryl’s mind turned like the vanes of a wind-powered turbine, chopping her sparrow-like thoughts into bloody pieces that fell onto a growing pile of forgotten memories."
#sentence="As the dark and mysterious stranger approached, Angela bit her lip anxiously, hoping with every nerve, cell, and fiber of her being that this would be the one man who would understand – who would take her away from all this – and who would not just squeeze her boob and make a loud honking noise, as all the others had."
sentence="I go to the store and I bought milk."
sentence="I will eat fish for dinner and drank milk with my dinner."
sentence="Matt like fish."
sentence="Anna and Mike is going skiing."

bin/token-activations.lua $model_file $index_file $data_file "$sentence"
