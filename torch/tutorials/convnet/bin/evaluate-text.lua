#!/usr/bin/env th

require 'convnet.utils';

local function makeLine(model, modelInfo, sentence)
  local output = torch.exp(model:forward(convertSentenceToIndices(sentence, modelInfo)))
  local m, i = torch.max(output, 1)
  local line = string.format('%d', i[1])
  for j=1,output:size(1) do
    line = line .. string.format(',%0.6f', output[j])
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
cmd:option('-header', false, 'include header in output')
cmd:text()

local opt = cmd:parse(arg or {})

local model, modelInfo = loadModelAndModelInfo(opt.modelPath, opt.modelInfoPath)

local lines = {}

if fileExists(opt.sentence) then
  f = io.open(opt.sentence, 'r')
  for sentence in io.lines(opt.sentence) do
    table.insert(lines, makeLine(model, modelInfo, sentence))
  end
else
  table.insert(lines, makeLine(model, modelInfo, sentence))
end

if opt.header then
  local noutput = model.modules[#model.modules-1].weight:size(1)
  local header = 'predicted'
  for j=1,noutput do
    header = header .. ',output' .. tostring(j)
  end
  print(header)
end

for i,line in ipairs(lines) do
  print(line)
end
