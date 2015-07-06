require 'hdf5';
require 'convnet.utils';
require 'convnet.inspect';

buildTokenActivations = function(tokens, posPolarity, negPolarity, posMagnitude, negMagnitude)
  local t = {}
  for i=1,#tokens do
    local tmp = {
      token=tokens[i],
      posPolarity=posPolarity[i],
      negPolarity=negPolarity[i],
      posMagnitude=posMagnitude[i],
      negMagnitude=negMagnitude[i]
    }
    table.insert(t, tmp)
  end
  return t
end

showTokenActivations = function(tokens, posPolarity, negPolarity, posMagnitude, negMagnitude)
  local tokActs = buildTokenActivations(
      tokens,
      posPolarity, negPolarity,
      posMagnitude, negMagnitude)
  for i=1,#tokActs do
    local output = string.format("%20s %2d %2d  %0.2f  %0.2f", 
      tokActs[i].token,
      tokActs[i].posPolarity, tokActs[i].negPolarity,
      tokActs[i].posMagnitude, tokActs[i].negMagnitude)
    print(output)
  end
end

-- Set model, model info, input, etc.
filterInfoFile = hdf5.open('results/test-wiki/okanohara/1.9m/filter-info-test-data.h5')
model = torch.load('results/test-wiki/okanohara/1.9m/model.net')
modelInfo = loadModelInfo('/tmp/sents-okanohara-wiki-dlm-train-initial-index.json')
-- sentence = 'In essence, to label a job, ability or duty as "practical" is limiting the potential of the one to whom it is assigned or chosen.'
sentence = 'What is a practical skill?'
input, tokens = convertSentenceToIndices(sentence, modelInfo)

recordingOpts = { activations=true, indices=true }
input2 = input:clone():resize(1, input:size(1))
pred, recording = predictAndRecordConvolution(model, input2, recordingOpts)

-- And go
filterInfo = filterInfoFile:read():all()
posPolarity, negPolarity = countPolarityOfActivations(recording, input2, filterInfo.positiveFilters, filterInfo.negativeFilters)
posMagnitude, negMagnitude = sumMagnitudeOfActivations(recording, input2, filterInfo.positiveFilters, filterInfo.negativeFilters)

showTokenActivations(tokens, posPolarity, negPolarity, posMagnitude, negMagnitude)
