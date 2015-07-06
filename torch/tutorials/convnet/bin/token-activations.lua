#!/usr/bin/env th

require 'hdf5';
require 'convnet.utils';
require 'convnet.inspect';

local buildTokenActivations = function(tokens, posPolarity, negPolarity, posMagnitude, negMagnitude)
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

local showTokenActivations = function(tokens, posPolarity, negPolarity, posMagnitude, negMagnitude)
  local tokActs = buildTokenActivations(
      tokens,
      posPolarity, negPolarity,
      posMagnitude, negMagnitude)
  for i=1,#tokActs do
    local output = string.format("%20s     %2d %2d  %0.2f  %0.2f", 
      tokActs[i].token,
      tokActs[i].posPolarity, tokActs[i].negPolarity,
      tokActs[i].posMagnitude, tokActs[i].negMagnitude)
    print(output)
  end
end


-- Function should take model, modelInfo, filterInfo, and sentence.
tokenActivations = function(model, modelInfo, filterInfo, sentence, opt)
  local input, tokens = convertSentenceToIndices(sentence, modelInfo)
  local output = model:forward(input)

  local recordingOpts = mergeTables(opt, { activations=true, indices=true })
  local input = input:clone():resize(1, input:size(1))
  local pred, recording = predictAndRecordConvolution(model, input, recordingOpts)

  local posPolarity, negPolarity = countPolarityOfActivations(recording, input, filterInfo.positiveFilters, filterInfo.negativeFilters)
  local posMagnitude, negMagnitude = sumMagnitudeOfActivations(recording, input, filterInfo.positiveFilters, filterInfo.negativeFilters)

  print('pred')
  print(pred)
  if output:nElement() > 1 then
    print('output')
    print(torch.exp(output))
  end
  showTokenActivations(tokens, posPolarity, negPolarity, posMagnitude, negMagnitude)
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify filters of a convnet')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'path to a serialized model')
cmd:argument('-modelInfo', 'path to the .json file containing metadata (vocabulary, etc.)')
cmd:argument('-filterInfo', 'path to file to which to write output')
cmd:argument('-sentence', 'sentence to be processed')
cmd:option('-gcFreq', 100, 'frequency of garbage collection calling kttorch.predict')
cmd:option('-verbose', false, 'print more information to terminal during execution')
cmd:text()

local opt = cmd:parse(arg or {})

-- model = torch.load('results/test-wiki/okanohara/1.9m/model.net')
-- modelInfo = loadModelInfo('/tmp/sents-okanohara-wiki-dlm-train-initial-index.json')
-- filterInfoFile = hdf5.open('results/test-wiki/okanohara/1.9m/filter-info-test-data.h5')

-- Set model, model info, input, etc.
local model = torch.load(opt.model)
local modelInfo = loadModelInfo(opt.modelInfo)
local filterInfoFile = hdf5.open(opt.filterInfo)
local filterInfo = filterInfoFile:read():all()

local sentence = 'What is a practical skill?'
tokenActivations(model, modelInfo, filterInfo, sentence, opt)
