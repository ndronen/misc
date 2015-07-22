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

local printTokenActivations = function(tokens, posPolarity, negPolarity, posMagnitude, negMagnitude, printLegend, printHeader)
  local tokActs = buildTokenActivations(
      tokens,
      posPolarity, negPolarity,
      posMagnitude, negMagnitude)

  if printHeader then
    if printLegend then
      print("Activations ['N' = number, 'S' = sum, 'acts' = activations, '+' = positive, '-' = negative]:")
    else
      print("Activations:")
    end

    local header = string.format("%20s     %10s %10s %8s %8s",
      "Token", "N+ acts.", "N- acts.",
      "S+ acts.", "S- acts.")
    print(header)
  end

  for i=1,#tokActs do
    local output = string.format("%20s     %10d %10d %8.2f %8.2f", 
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

  local numPrints = opt.pred and 1 or 0
  numPrints = numPrints + (opt.output and 1 or 0) + (opt.activations and 1 or 0)
  local printHeader = numPrints > 1

  if opt.pred then
    if printHeader then
      print("Pred: " .. pred[1])
    else
      print(pred[1])
    end
  end

  if opt.output then
    if output:nElement() > 1 then
      local output = torch.exp(output)
      local line = ''
      for i=1,output:size(1) do
        line = line .. output[i]
        if i < output:size(1) then
          line = line .. ','
        end
      end
      if printHeader then
        print("Output: " .. line)
      else
        print(line)
      end
    end
  end

  if opt.activations then
    printTokenActivations(tokens,
      posPolarity, negPolarity,
      posMagnitude, negMagnitude,
      opt.verbose, printHeader)
  end
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Get prediction, softmax output, and per-token activation information for sentences')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'path to a serialized model')
cmd:argument('-modelInfo', 'path to the .json file containing metadata (vocabulary, etc.)')
cmd:argument('-filterInfo', 'path to file to which to write output')
cmd:argument('-sentence', 'sentence to be processed')
cmd:option('-gcFreq', 100, 'frequency of garbage collection calling kttorch.predict')
cmd:option('-verbose', false, 'print more information to terminal during execution')
cmd:option('-pred', false, "print the model's prediction")
cmd:option('-output', false, "print the model's (softmax) output")
cmd:option('-activations', false, "print per-token activations")
cmd:text()

local opt = cmd:parse(arg or {})

-- Set model, model info, input, etc.
local model = loadModel(opt.model)
local modelInfo = loadModelInfo(opt.modelInfo)
local filterInfoFile = hdf5.open(opt.filterInfo)
local filterInfo = filterInfoFile:read():all()

if fileExists(opt.sentence) then
  local f = io.open(opt.sentence, 'r')
  for sentence in io.lines(opt.sentence) do
    tokenActivations(model, modelInfo, filterInfo, sentence, opt)
  end
else
  tokenActivations(model, modelInfo, filterInfo, opt.sentence, opt)
end
