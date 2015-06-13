require 'kttorch'
require 'cutorch'
require 'fbcunn'
require('fb.luaunit')
require 'hdf5'
local torch = require('fbtorch')

if not opt then
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
  cmd:option('-zeroVector', 107701, 'index of zero vector in dictionary: [1, dict size]; 0 means there is no zero vector')
  cmd:option('-wordDims', 50, 'number of dimensions of word representations')
  cmd:option('-word2Vec', false, 'use pretrained word2vec weights in lookup table')
  cmd:option('-fixWords', false, 'disable updates of word representations')
  cmd:option('-maxWordNorm', 20, 'maximum 2-norm of word representations in lookup table')
  cmd:option('-spatial', false, 'train a spatial convolutional network')
  cmd:option('-activation', 'relu', 'activation function: relu | tanh')
  cmd:text()
  opt = cmd:parse(arg or {})
end

if string.match(opt.nFullyConnectedLayers, ",") then
  local layerSizeStrings = string.split(opt.nFullyConnectedLayers, ",")
  opt.nFullyConnectedLayers = {}
  for i, size in ipairs(layerSizeStrings) do
    table.insert(opt.nFullyConnectedLayers, tonumber(size))
  end
else
  opt.nFullyConnectedLayers = { tonumber(opt.nFullyConnectedLayers) }
end

local activation = nil
if opt.activation == 'relu' then
  activation = nn.ReLU
elseif opt.activation == 'tanh' then
  activation = nn.Tanh
end

nWords = 107701

inputFrameSize = opt.wordDims; -- dimensionality of one sequence element 
kw = opt.kernelWidth;          -- kernel spans three input elements
dw = 1;          -- we step once and go on to the next sequence element

-- K for k-max pooling
k = 1

-- a typical modern convolution network (conv+relu+pool)
model = nn.Sequential()
renormer = kttorch.Renormer()

if opt.zeroVector ~= 0 then
  model:add(kttorch.LookupTableInputZeroer(0.2, opt.zeroVector))
end

local lookupTable = nil
if opt.type == 'cuda' then
  if opt.lookupOnGpu then
    lookupTable = nn.LookupTableGPU(nWords, opt.wordDims)
  else
    lookupTable = nn.LookupTable(nWords, opt.wordDims)
  end
else
  lookupTable = nn.LookupTable(nWords, opt.wordDims)
end

if opt.word2Vec then
  word2VecFile = hdf5.open('okanohara-weights.h5', 'r')
  word2VecData = word2VecFile:read():all()
  lookupTable.weight:copy(word2VecData.weight)
end

if opt.zeroVector ~= 0 then
  lookupTable.weight[opt.zeroVector]:zero()
end

if opt.fixWords then
  lookupTable = kttorch.FixedLookupTable(lookupTable)
end

lookupTable.weight:renorm(2, 1, opt.maxWordNorm)
model:add(lookupTable)

if opt.spatial then
  -- This doesn't work yet.
  model:add(nn.SpatialConvolution(inputFrameSize, opt.nKernels, kw, kw))
  model:add(activation())
  model:add(nn.SpatialMaxPooling(kw, kw))

  model:add(nn.SpatialConvolution(opt.nKernels, 5, kw, kw))
  model:add(activation())
  model:add(nn.SpatialMaxPooling(kw, kw))
else
  -- if opt.type == 'cuda' then
  --  model:add(nn.TemporalConvolutionFB(inputFrameSize, opt.nKernels, kw, dw))
  -- else
  conv = nn.TemporalConvolution(inputFrameSize, opt.nKernels, kw, dw)
  renormer:add(conv, opt.maxNorm)
  model:add(conv)

  model:add(activation())
  -- end

  if k == 1 then
    -- Is it (width - kernel width) or (width - (kernel width - 1))?
    -- model:add(nn.TemporalMaxPooling(trainData.data:size(2)-(opt.kernelWidth-1)))
    model:add(nn.TemporalMaxPooling(trainData.data:size(2)-opt.kernelWidth))
  else
    model:add(nn.TemporalKMaxPooling(k))
  end
end

-- model:add(nn.BatchNormalization(0))
model:add(nn.Dropout(0.5))
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
  -- model:add(nn.BatchNormalization(0))
  linear = nn.Linear(ninput, noutput)
  renormer:add(linear, opt.maxNorm)
  penultimateOutput = noutput
  model:add(linear)
  model:add(activation())
  model:add(nn.Dropout(0.5))
end

-- stage 4: output layer (before cost)
local noutputs = nil
if opt.loss == 'mse' then
  noutputs = 1
else
  noutputs = torch.max(trainData.labels)
end

-- model:add(nn.BatchNormalization(0))
model:add(nn.Linear(penultimateOutput, noutputs))

if opt.type == 'cuda' then
  model:cuda()
end
