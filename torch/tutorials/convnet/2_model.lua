----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
  print '==> processing options'
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('SVHN Model Definition')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-visualize', true, 'visualize input data and weights during training')
  cmd:option('-fullyConnectedLayers', 1, 'number of extra fully-connected layers after convolutional layers')
  cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
  cmd:text()
  opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 2-class problem
if opt.loss == 'mse' then
  noutputs = 1
  nonlinearity = nn.Tanh
else
  noutputs = torch.max(trainData.labels)
  nonlinearity = nn.ReLU
end

--[[
print '==> sentence length'
print(trainData.data:size(2))
--]]

nWords = 107701
nWordDims = 50
inputFrameSize = nWordDims; -- dimensionality of one sequence element 
outputFrameSize = 500;       -- number of derived features for one sequence element
kw = 3;          -- kernel spans three input elements
dw = 1;          -- we step once and go on to the next sequence element

----------------------------------------------------------------------
print '==> construct model'

-- a typical modern convolution network (conv+relu+pool)
model = nn.Sequential()

-- stage 1: lookup table
model:add(nn.LookupTable(nWords, nWordDims))

--[[
print '==> nn.TemporalConvolution'
print 'inputFrameSize'
print(inputFrameSize)
print 'outputFrameSize'
print(outputFrameSize)
print 'kw'
print(kw)
print 'dw'
print(dw)
--]]

-- stage 2: filter bank -> squashing -> pooling
model:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kw, dw))
model:add(nn.ReLU())
-- model:add(nonlinearity())
model:add(nn.TemporalMaxPooling(trainData.data:size(2)-2))
model:add(nn.Dropout(0.5))
model:add(nn.View(outputFrameSize))

-- stage 3: fully-connected layers
for i = 1,opt.fullyConnectedLayers do
  local nunits = 0
  if i == 1 then
    nunits = outputFrameSize
  else
    nunits = 50
  end
  model:add(nn.Linear(nunits, 50))
  model:add(nn.ReLU())
  -- model:add(nonlinearity())
  model:add(nn.Dropout(0.5))
end

-- stage 4: output layer (before cost)
if opt.fullyConnectedLayers < 1 then
  model:add(nn.Linear(outputFrameSize, noutputs))
else
  model:add(nn.Linear(50, noutputs))
end

-- print '==> here is the model:'
-- print(model)

-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
