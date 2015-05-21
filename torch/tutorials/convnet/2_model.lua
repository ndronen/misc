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

-- require 'torch'   -- torch
-- require 'nn'      -- provides all sorts of trainable modules/layers

require 'cutorch'
require 'fbcunn'
require('fb.luaunit')
local torch = require('fbtorch')

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
  cmd:option('-type', 'double', 'type: double | float | cuda')
  cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
  cmd:option('-kernelWidth', 3, 'width of kernels: 2 or greater')
  cmd:option('-nKernels', 500, 'number of kernels: 2 or greater')
  cmd:option('-nFullyConnectedLayers', 1, 'number of extra fully-connected layers after convolutional layers')
  cmd:option('-lookupOnGpu', true, 'put the lookup table on the GPU')
  cmd:text()
  opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 2-class problem
if opt.loss == 'mse' then
  noutputs = 1
else
  noutputs = torch.max(trainData.labels)
end

nWords = 107701
nWordDims = 50

inputFrameSize = nWordDims; -- dimensionality of one sequence element 
kw = 3;          -- kernel spans three input elements
dw = 1;          -- we step once and go on to the next sequence element

----------------------------------------------------------------------
print '==> construct model'
-- K for k-max pooling
k = 1

-- a typical modern convolution network (conv+relu+pool)
model = nn.Sequential()

-- stage 1: lookup table
if opt.type == 'cuda' then
  if opt.lookupOnGpu then
    model:add(nn.LookupTableGPU(nWords, nWordDims, true))
  else
    model:add(nn.LookupTable(nWords, nWordDims))
  end
else
  model:add(nn.LookupTable(nWords, nWordDims))
end

-- if opt.type == 'cuda' then
--  model:add(nn.TemporalConvolutionFB(inputFrameSize, opt.nKernels, kw, dw))
-- else
conv = nn.TemporalConvolution(inputFrameSize, opt.nKernels, kw, dw)
-- conv.bias = torch.ones(conv.bias:size())
model:add(conv)
-- end
model:add(nn.ReLU())

if k == 1 then
  model:add(nn.TemporalMaxPooling(trainData.data:size(2)-2))
else
  model:add(nn.TemporalKMaxPooling(k))
end

model:add(nn.Dropout(0.5))
model:add(nn.View(k * opt.nKernels))

-- stage 3: fully-connected layers
for i = 1,opt.nFullyConnectedLayers do
  local nunits = opt.nKernels
  if i == 1 then
    nunits = k * opt.nKernels
  end
  linear = nn.Linear(nunits, nunits)
  -- linear.bias = torch.ones(linear.bias:size())
  model:add(linear)
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
end

-- stage 4: output layer (before cost)
local last = nil
if opt.nFullyConnectedLayers < 1 then
  last = nn.Linear(opt.nKernels, noutputs)
else
  last = nn.Linear(opt.nKernels, noutputs)
end
last.bias = torch.ones(last.bias:size())
model:add(last)

if opt.type == 'cuda' then
  model:cuda()
end
