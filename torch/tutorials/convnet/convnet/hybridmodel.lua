buildHybridModel = function(opt)

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
  local kw = opt.filterWidth;          -- kernel spans three input elements
  local dw = 1;          -- we step once and go on to the next sequence element
  
  -- K for k-max pooling
  local k = 1
  
  -- a typical modern convolution network (conv+relu+pool)
  local convnet = nn.Sequential()
  -- a simple network that takes a word index as input and output a word embedding.
  local word = nn.Sequential()
  -- an even simpler network that takes a number as input and outputs it.
  local position = nn.Sequential()
  position:add(nn.Identity())

  local renormers = kttorch.Renormers()

  -- TODO: change this to opt.zeroVector ~= nil
  if opt.zeroVector > 0 then
    convnet:add(kttorch.SentenceCompacter(opt.padding, opt.zeroVector))
  end
  
  if opt.convnetLookupTableDropout > 0 and opt.zeroVector ~= 0 then
    convnet:add(kttorch.LookupTableInputZeroer(opt.convnetLookupTableDropout, opt.zeroVector))
  end
  
  local convnetLookupTable = nil
  
  if opt.type == 'cuda' then
    convnetLookupTable = nn.LookupTableGPU(opt.nWords, opt.wordDims)
  else
    convnetLookupTable = nn.LookupTable(opt.nWords, opt.wordDims)
  end
  
  if opt.word2Vec then
    error('TODO: add option for word2vec file')
    word2VecFile = hdf5.open('okanohara-weights.h5', 'r')
    word2VecData = word2VecFile:read():all()
    convnetLookupTable.weight:copy(word2VecData.weight)
  end
  
  if opt.zeroZeroVector then
    convnetLookupTable.weight[opt.zeroVector]:zero()
  end
  
  if opt.fixWords then
    convnetLookupTable = kttorch.FixedLookupTable(convnetLookupTable)
  end

  --[[
  Use LookupTableRenormer to make the l2 norms of the word representations
  uniform before training.
  --]]
  local wordRenormer = nil 
  local resetVal = 10

  if opt.maxWordNorm == 0 then
    wordRenormer = kttorch.LookupTableRenormer(convnetLookupTable, 1)
  else
    wordRenormer = kttorch.LookupTableRenormer(convnetLookupTable, opt.maxWordNorm)
    -- Not sure if this does the right thing.
    resetVal = opt.maxWordNorm * 2
  end

  if opt.forceWordsOnBall then
    -- A bit of a hack.  Make all of the weights big so the L2 norms are greater
    -- than the max.  This forces all words to have the same norm after renorming.
    convnetLookupTable:reset(resetVal)
  end

  wordRenormer:renorm()

  if opt.renormFreq > 0 then
    renormers:add(wordRenormer)
  end

  -- Share the word embeddings.
  convnet:add(convnetLookupTable)
  word:add(convnetLookupTable:clone('weight'))

  local penultimateOutput = k * opt.nFilters
  local ninput = penultimateOutput
  
  local conv = nn.TemporalConvolution(inputFrameSize, opt.nFilters, kw, dw)
  --[[
  Only renorm the convolutional filters if explicitly requested.
  This is unlike the lookup table, where I renorm before training
  even if the word representations won't be renormed during training.
  I haven't been renorming the filters before training, and I don't
  want to try something new just yet.
  --]]
  if opt.maxFilterNorm > 0 then
    if opt.forceFiltersOnBall then
      --[[
      A bit of a hack.  Make all of the weights big so the L2 norms
      are greater than the max.  This forces all words to have the
      same norm after renorming.
      --]]
      conv:reset(10)
    end
    local convRenormer = kttorch.TemporalConvolutionRenormer(
        conv, opt.maxFilterNorm)
    convRenormer:renorm()
    renormers:add(convRenormer)
  end
  convnet:add(conv)
  convnet:add(activation())

  -- Use TemporalKMaxPooling instead of TemporalMaxPooling.  The problem
  -- with TemporalMaxPooling is that it requires a kernel size, which
  -- is complicated when using kttorch.SentenceCompacter to shrink
  -- sentences to their natural length (and not the heavily zero-padded,
  -- fixed-width length that occurs when you put them into a matrix format.
  convnet:add(nn.TemporalKMaxPooling(k))
  convnet:add(nn.BatchNormalization(k * opt.nFilters))
  convnet:add(nn.View(k * opt.nFilters))

  -- Need to combine convnet, word, and position models.
  local model = nn.Sequential()
  local parallel = nn.ParallelTable()
  local join = nn.JoinTable(1)

  parallel:add(convnet)
  parallel:add(word)
  parallel:add(position)

  model:add(parallel)
  model:add(join)

  -- stage 3: fully-connected layers
  for i,noutput in ipairs(opt.nFullyConnectedLayers) do
    if i > 1 then
      ninput = opt.nFullyConnectedLayers[i-1]
    end
    local linear = nn.Linear(ninput, noutput)
    penultimateOutput = noutput
    model:add(linear)
    model:add(activation())
    model:add(nn.BatchNormalization(noutput))
  end
  
  -- stage 4: output layer (before cost)
  local noutputs = nil
  if opt.loss == 'mse' then
    noutputs = 1
  else
    noutputs = torch.max(trainData.labels)
  end
  
  model:add(nn.Linear(penultimateOutput, noutputs))

  return model, renormers
end
