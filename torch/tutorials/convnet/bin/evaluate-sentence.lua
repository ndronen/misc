#!/usr/bin/env th

require 'convnet.utils';


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict using a saved model')
cmd:text()
cmd:text('Options:')
cmd:argument('-modelPath', 'path to the model')
cmd:argument('-modelInfoPath', 'path to the .json file containing metadata (vocabulary, etc.)')
cmd:argument('-s', 'the sentence to evaluate')
cmd:text()

local opt = cmd:parse(arg or {})

local model, modelInfo = loadModelAndModelInfo(opt.modelPath, opt.modelInfoPath)

local output = torch.exp(model:forward(convertSentenceToIndices(opt.s, modelInfo)))

local header = 'max,predicted class,'
for j=1,output:nElement() do
  header = header .. ',softmax' .. tostring(i)
end

local m, i = torch.max(output, 1)

print(header)

local line = string.format('%0.6f,%d', m[1], i[1])
for j=1,output:nElement() do
  line = line .. string.format(',%0.6f', output[j])
end
print(line)
