#!/usr/bin/env th

require 'convnet.utils';

local function makeLine(model, modelInfo, sentence, opt)
  local output = torch.exp(model:forward(convertSentenceToIndices(sentence, modelInfo)))
  local m, i = torch.max(output, 1)
  local line = string.format('%d', i[1])
  for j=1,output:size(1) do
    line = line .. string.format(opt.sep .. '%0.6f', output[j])
  end
  if opt.printSentence then
    line = line .. opt.sep .. sentence
  end
  return line
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Get model prediction(s) given text input(s)')
cmd:text()
cmd:text('Options:')
cmd:argument('-modelPath', 'path to the model')
cmd:argument('-modelInfoPath', 'path to the .json file containing metadata (vocabulary, etc.)')
cmd:argument('-sentence', 'a sentence or a path to a file containing one sentence per line')
cmd:option('-printHeader', false, 'include header in output')
cmd:option('-printSentence', false, 'include sentence in output')
cmd:option('-sep', '\t', 'separator for output')
cmd:text()

local opt = cmd:parse(arg or {})

local model, modelInfo = loadModelAndModelInfo(opt.modelPath, opt.modelInfoPath)

local lines = {}

if fileExists(opt.sentence) then
  f = io.open(opt.sentence, 'r')
  for sentence in io.lines(opt.sentence) do
    table.insert(lines, makeLine(model, modelInfo, sentence, opt))
  end
else
  table.insert(lines, makeLine(model, modelInfo, opt.sentence, opt))
end

if opt.printHeader then
  local noutput = model.modules[#model.modules-1].weight:size(1)
  local header = 'predicted'
  for j=1,noutput do
    header = header .. opt.sep .. 'output' .. tostring(j)
  end
  if opt.printSentence then
    header = header .. opt.sep .. 'sentence'
  end
  print(header)
end

for i,line in ipairs(lines) do
  print(line)
end
