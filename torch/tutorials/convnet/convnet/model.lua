buildModel = function(opt)

  if opt.type == 'float' then
    require 'nn'
    torch.setdefaulttensortype('torch.FloatTensor')
  elseif opt.type == 'double' then
    require 'nn'
    torch.setdefaulttensortype('torch.DoubleTensor')
  elseif opt.type == 'cuda' then
    require 'fbcunn'
    torch = require 'fbtorch'
    torch.setdefaulttensortype('torch.FloatTensor')
  end

  require 'kttorch'

  opt.nWords = tonumber(opt.nWords)
  
  if string.match(opt.nFullyConnectedLayers, ",") then
    local layerSizeStrings = string.split(opt.nFullyConnectedLayers, ",")
    opt.nFullyConnectedLayers = {}
    for i, size in ipairs(layerSizeStrings) do
      table.insert(opt.nFullyConnectedLayers, tonumber(size))
    end
  else
    if opt.nFullyConnectedLayers ~= "0" then
      opt.nFullyConnectedLayers = { tonumber(opt.nFullyConnectedLayers) }
    else 
      opt.nFullyConnectedLayers = {}
    end
  end
  
  local activation = nil
  if opt.activation == 'relu' then
    activation = nn.ReLU
  elseif opt.activation == 'tanh' then
    activation = nn.Tanh
  end
  
  local inputFrameSize = opt.wordDims; -- dimensionality of one sequence element 
  local kw = opt.kernelWidth;          -- kernel spans three input elements
  local dw = 1;          -- we step once and go on to the next sequence element
  
  -- K for k-max pooling
  local k = 1
  
  -- a typical modern convolution network (conv+relu+pool)
  local model = nn.Sequential()
  local renormer = kttorch.Renormer()
  
  if opt.lookupTableDropout > 0 and opt.zeroVector ~= 0 then
    model:add(kttorch.LookupTableInputZeroer(opt.lookupTableDropout, opt.zeroVector))
  end
  
  local lookupTable = nil
  
  if opt.type == 'cuda' then
    lookupTable = nn.LookupTableGPU(opt.nWords, opt.wordDims)
  else
    lookupTable = nn.LookupTable(opt.nWords, opt.wordDims)
  end
  
  if opt.word2Vec then
    error('TODO: add option for word2vec file')
    word2VecFile = hdf5.open('okanohara-weights.h5', 'r')
    word2VecData = word2VecFile:read():all()
    lookupTable.weight:copy(word2VecData.weight)
  end
  
  if opt.zeroZeroVector then
    lookupTable.weight[opt.zeroVector]:zero()
  end
  
  if opt.fixWords then
    lookupTable = kttorch.FixedLookupTable(lookupTable)
  end
  
  if opt.type == 'cuda' then
    lookupTable.weight = lookupTable.weight:clone():float():renorm(2, 1, opt.maxWordNorm)
  else
    lookupTable.weight:renorm(2, 1, opt.maxWordNorm)
  end
  model:add(lookupTable)
  
  local penultimateOutput = k * opt.nKernels
  local ninput = penultimateOutput
  
  if opt.spatial then
    model:add(nn.View(-1, trainData.data:size(2), opt.wordDims))
  
    -- A spatial convolution can take either grayscale (i.e. single-channel)
    -- or multi-channel (e.g. RGB, YUV) images.  When the input is a sentence
    -- matrix, there is only a single channel; hence, the first argument to
    -- the first layer's nn.SpatialConvolution(MM) is fixed to 1.
    local l1NFilters = 3
    local l1kW = 15
    local l1kH = 5
    -- model:add(kttorch.InputPrinter('SpatialConvolutionMM #1'))
    model:add(nn.SpatialConvolutionMM(1, l1NFilters, l1kW, l1kH))
    model:add(activation())
    -- expected output: 5 X 41 X 11
    -- actual output:   5 X 41 X 11
  
    local l1PoolW = 2
    local l1PoolH = 5
    -- model:add(kttorch.InputPrinter('SpatialMaxPooling #1'))
    model:add(nn.SpatialMaxPooling(l1PoolW, l1PoolH, 1, 1))
    -- expected output: 5 X 37 X 10
    -- actual output:   5 X 37 X 10
  
    local l2NFilters = 15
    local l2kW = 5
    local l2kH = 10
    -- model:add(kttorch.InputPrinter('SpatialConvolutionMM #2'))
    model:add(nn.SpatialConvolutionMM(l1NFilters, l2NFilters, l2kW, l2kH))
    model:add(activation())
    -- These sizes don't seem right.  Shouldn't the number of feature maps
    -- output by this convolution be #feature maps from previous layer X
    -- number of filters in this layer?
    -- expected output: 15 X 28 X 6
    -- actual output:  15 X 28 X 6
  
    local l2PoolW = 3
    local l2PoolH = 9
    -- model:add(kttorch.InputPrinter('SpatialMaxPooling #2'))
    model:add(nn.SpatialMaxPooling(l2PoolW, l2PoolH, 1, 1))
    -- output 15 X 20 X 4
  
    -- I this this might have become 15 parallel temporal convolutions
    -- taking 2D inputs of size 20 X 5.
    p = nn.Parallel(1, 1)
    for i=1,l2NFilters do
      local tNInputs = 4
      local tNFilters = 1
      local tKW = 3
      s = nn.Sequential()
      -- s:add(kttorch.InputPrinter('TemporalConvolution #' .. i))
      s:add(nn.TemporalConvolution(tNInputs, tNFilters, tKW))
      -- output 20 X 4
      -- s:add(kttorch.InputPrinter('TemporalMaxPooling #' .. i))
      s:add(nn.TemporalMaxPooling(18))
  
      p:add(s)
    end
    model:add(p)
  
    -- model:add(nn.Dropout(0.5))
    -- model:add(kttorch.InputPrinter('View (output of Parallel)'))
    ninput = l2NFilters
    penultimateOutput = l2NFilters
    model:add(nn.View(penultimateOutput))
  else
    -- if opt.type == 'cuda' then
    --  model:add(nn.TemporalConvolutionFB(inputFrameSize, opt.nKernels, kw, dw))
    -- else
    local conv = nn.TemporalConvolution(inputFrameSize, opt.nKernels, kw, dw)
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
    -- model:add(nn.BatchNormalization(0))
    model:add(nn.Dropout(0.5))
    model:add(nn.View(k * opt.nKernels))
  end
  
  -- stage 3: fully-connected layers
  
  for i,noutput in ipairs(opt.nFullyConnectedLayers) do
    if i > 1 then
      ninput = opt.nFullyConnectedLayers[i-1]
    end
    -- model:add(nn.BatchNormalization(0))
    local linear = nn.Linear(ninput, noutput)
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
  -- model:add(kttorch.InputPrinter('Final Linear'))
  model:add(nn.Linear(penultimateOutput, noutputs))
  
  return model, renormer
end