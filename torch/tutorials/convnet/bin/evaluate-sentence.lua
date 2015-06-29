#!/usr/bin/env th

require 'convnet.utils';

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict using a saved model')
cmd:text()
cmd:text('Options:')
cmd:argument('-modelPath', 'path to the model')
cmd:argument('-modelInfoPath', 'path to the .json file containing metadata (vocabulary, etc.)')
cmd:argument('-sentence', 'the sentence to evaluate')
cmd:option('-header', false, 'include header in output')
cmd:text()

local opt = cmd:parse(arg or {})

local model, modelInfo = loadModelAndModelInfo(opt.modelPath, opt.modelInfoPath)

local output = torch.exp(model:forward(convertSentenceToIndices(opt.sentence, modelInfo)))

if opt.header then
  local header = 'predicted'
  for j=1,output:nElement() do
    header = header .. ',softmax' .. tostring(j)
  end
  print(header)
end
local m, i = torch.max(output, 1)

local line = string.format('%d', i[1])
for j=1,output:nElement() do
  line = line .. string.format(',%0.6f', output[j])
end

print(line)
