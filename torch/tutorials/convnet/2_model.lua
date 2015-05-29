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

require 'kttorch'
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
  cmd:option('-nFullyConnectedLayers', '1', 'number of extra fully-connected layers after convolutional layers')
  cmd:option('-lookupOnGpu', true, 'put the lookup table on the GPU')
  cmd:option('-zeroVector', 107701, 'index of zero vector in dictionary: [1, dict size]')
  cmd:text()
  opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

if string.match(opt.nFullyConnectedLayers, ",") then
  local layerSizeStrings = string.split(opt.nFullyConnectedLayers, ",")
  opt.nFullyConnectedLayers = {}
  for i, size in ipairs(layerSizeStrings) do
    table.insert(opt.nFullyConnectedLayers, tonumber(size))
  end
else
  opt.nFullyConnectedLayers = { tonumber(opt.nFullyConnectedLayers) }
end

nWords = 107701
nWordDims = 50

inputFrameSize = nWordDims; -- dimensionality of one sequence element 
kw = opt.kernelWidth;          -- kernel spans three input elements
dw = 1;          -- we step once and go on to the next sequence element

----------------------------------------------------------------------
print '==> construct model'
-- K for k-max pooling
k = 1

-- a typical modern convolution network (conv+relu+pool)
model = nn.Sequential()

model:add(kttorch.LookupTableInputZeroer(0.2, opt.zeroVector))

-- stage 1: lookup table
local lookupTable = nil
if opt.type == 'cuda' then
  if opt.lookupOnGpu then
    lookupTable = nn.LookupTableGPU(nWords, nWordDims, true)
  else
    lookupTable = nn.LookupTable(nWords, nWordDims)
  end
else
  lookupTable = nn.LookupTable(nWords, nWordDims)
end
lookupTable.weight[opt.zeroVector]:zero()
model:add(lookupTable)

-- if opt.type == 'cuda' then
--  model:add(nn.TemporalConvolutionFB(inputFrameSize, opt.nKernels, kw, dw))
-- else
model:add(nn.TemporalConvolution(inputFrameSize, opt.nKernels, kw, dw))
-- end
model:add(nn.ReLU())

if k == 1 then
  -- Is it (width - kernel width) or (width - (kernel width - 1))?
  -- model:add(nn.TemporalMaxPooling(trainData.data:size(2)-(opt.kernelWidth-1)))
  model:add(nn.TemporalMaxPooling(trainData.data:size(2)-opt.kernelWidth))
else
  model:add(nn.TemporalKMaxPooling(k))
end

-- model:add(nn.Dropout(0.5))
model:add(nn.View(k * opt.nKernels))

-- stage 3: fully-connected layers
local penultimateOutput = nil
for i,noutput in ipairs(opt.nFullyConnectedLayers) do
  local ninput = nil
  if i == 1 then
    ninput = k * opt.nKernels
  else
    ninput = opt.nFullyConnectedLayers[i-1]
  end
  linear = nn.Linear(ninput, noutput)
  penultimateOutput = noutput
  model:add(linear)
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
end

-- stage 4: output layer (before cost)
local noutputs = nil
if opt.loss == 'mse' then
  noutputs = 1
else
  noutputs = torch.max(trainData.labels)
end

model:add(nn.Linear(penultimateOutput, noutputs))

if opt.type == 'cuda' then
  model:cuda()
end
