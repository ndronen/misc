require 'hdf5'
require 'randomkit'

--[[
By default, require all indices in a permutation to be other than the original sequence.
--]]
shuffleIndices = function(size, opts) 
  local opts = opts or { requireFullShuffle=true }
  local indices = nil

  if size == 1 then
    return torch.Tensor({ size })
  end

  while true do
    indices = torch.randperm(size)
    if not opts.requireFullShuffle then
      break
    end

    local ok = true
    for i=1,indices:size(1) do
      if indices[i] == i then
        ok = false
      end
    end
    if ok then break end
  end
  return indices
end

--[[
--]]
makePermutationNegativeExamples = function(data, labels, opts)
  data = data:clone()
  labels = labels:clone()

  local opts = opts or {}
  local negativeLabel = opts.negativeLabel or 1
  local padding = opts.padding or 2
  local lengths = opts.lengths or error('lengths is required')

  if opts.noiseCount <= 0 and opts.noiseFraction <= 0.0 then
    error("either noiseCount or noiseFraction is required in 'opts'")
  end

  -- Find examples that have the negative class's label, place
  -- the permutation window randomly on the sentence, and permute
  -- the words in the window.
  for i=1,data:size(1) do
    if labels[i] == negativeLabel then
      local windowSize = opts.noiseCount
      if windowSize == 0 then
        windowSize = math.ceil(lengths[i] * opts.noiseFraction)
      end
      local maxWindowStartOffset = lengths[i] - windowSize + 1
      local windowStartIndex = padding + torch.random(maxWindowStartOffset) 
      local values = data[i]:narrow(1, windowStartIndex, windowSize):clone()
      local shuffledIndices = shuffleIndices(windowSize) + windowStartIndex - 1
      for j=1,windowSize do
        data[i][shuffledIndices[j]] = values[j]
      end
    end
  end

  return data, labels
end

--[[
--]]
extractKeysFromTable = function(set)
  local keys = {}
  local i = 1

  for k,v in pairs(set) do
    keys[i] = k
    i = i+1
  end

  return keys
end

--[[
--]]
sampleReplacementIndices = function(numToReplace, low, high)
  local replacementSet = {}
  local numTries = 0
  while #extractKeysFromTable(replacementSet) < numToReplace do
    if numTries > numToReplace * 10 then
      error("not able to find enough replacement indices;" ..
        " numTries " .. numTries ..
        " numToReplace " .. numToReplace ..
        " low " .. low ..
        " high " .. high)
    end
    numTries = numTries + 1

    local replacementIndex = torch.random(low, high)
    replacementSet[replacementIndex] = true
  end

  return extractKeysFromTable(replacementSet)
end

--[[
--]]
makeReplacementNegativeExamples = function(data, labels, opts)
  data = data:clone()
  labels = labels:clone()

  local opts = opts or {}
  local negativeLabel = opts.negativeLabel or 1
  local padding = opts.padding or 2
  local lengths = opts.lengths or error('lengths is required')

  if opts.noiseCount <= 0 and opts.noiseFraction <= 0.0 then
    error("either noiseCount or noiseFraction is required in 'opts'")
  end

  local maxIndex = opts.maxIndex or torch.max(data)

  -- Find examples that have the negative class's label and replace
  -- one word with a random word.  A word here is an index into the
  -- vocabulary.
  for i=1,data:size(1) do
    if labels[i] == negativeLabel then
      local numToReplace = opts.noiseCount
      if numToReplace == 0 then
        numToReplace = math.ceil(lengths[i] * opts.noiseFraction)
      end

      local low = padding + 1
      local high = data:size(2) - padding
      local replacementIndices = sampleReplacementIndices(numToReplace, low, high)

      for j=1,#replacementIndices do
        local wordToReplace = replacementIndices[j]
        local currentWord = data[i][wordToReplace]
        local newWord = currentWord
        while newWord == currentWord do
          newWord = torch.random(maxIndex)
          data[i][wordToReplace] = newWord
        end
      end
    end
  end
  return data, labels
end

--[[
--]]
filterByLength = function(data, labels, length, direction, zeroVectorIndex, padding)
  local numZero = countZeroesInSentence(data, zeroVectorIndex, padding)
  local numWords = -numZero + data:size(2)
  local mask = nil
  if direction == 'min' then
    mask = numWords:gt(length):byte()
  else
    mask = numWords:lt(length):byte()
  end
  local indices = torch.linspace(1, mask:size(1), mask:size(1)):long()
  local selected = indices[mask:eq(1)]
  local selectedLabels = labels:index(1, selected)
  local selectedData = data:index(1, selected)
  return selectedData, selectedLabels
end

--[[
--]]
countZeroesInSentence = function(data, zeroVectorIndex, padding)
  -- Make a binary mask of the unknown or zero-padding words in each sentence.
  zeroVectorMask = data:eq(zeroVectorIndex):int()

  --[[
  Count the unknown and zero-padding words in each sentence by summing
  along the rows, then subtract the number of zero-padding words,
  since they don't add to the length of the sentence.
  --]]
  return torch.sum(zeroVectorMask, 2) - 2*padding
end

--[[
Input labels are 0 for negative example, 1 for positive example.
Transform these to (max length-num words) for negative examples and (max
length+num words) for positive examples.  This should cause the model
to try to place extremely short positive and negative examples near one
another and to place extremely long positive and negative examples far
from one another.
--]]
scaleRegressionTargets = function(dataset, zeroVectorIndex, padding)
  numZero = countZeroesInSentence(dataset.X, zeroVectorIndex, padding)
  numZero = numZero:reshape(numZero:nElement())
  numZero[torch.lt(numZero, 1)] = 1

  --[[
  Make a mask of the two types of examples and use them to update the
  regression targets.
  --]]
  negMask = dataset.y:eq(1)
  maxLength = dataset.X:size(2) - 2*padding - 1

  --[[
  A conditional function:
    target = maxLength - length if negative example
    target = maxLength + length if positive example
  --]]
  regTargets = numZero
  regTargets[negMask] = -regTargets[negMask]
  regTargets = regTargets + maxLength

  -- Rescale.
  regTargets = torch.round((regTargets:float()/opt.scaleMseTarget))+1
  return regTargets
end

loadData = function(opt) 
  -- Load train/test data.
  if not fileExists(opt.trainFile) then
    error('Train data file does not exist ' .. opt.trainFile)
  end
  
  local trainHdfFile = hdf5.open(opt.trainFile, 'r')
  local trainHdfData = trainHdfFile:read():all()
  
  local testHdfFile = nil
  local testHdfData = nil
  if opt.test then
    if not fileExists(opt.testFile) then
      error('Test data file does not exist ' .. opt.testFile)
    end
    testHdfFile = hdf5.open(opt.testFile, 'r')
    testHdfData = testHdfFile:read():all()
  end
  
  -- Optionally scale the target variable if the loss is mean squared error.
  if (opt.loss == 'mse') and (opt.scaleMseTarget) then
    trainHdfData.y = scaleRegressionTargets(
        trainHdfData, opt.zeroVector, opt.padding)
    if opt.test then
      testHdfData.y = scaleRegressionTargets(
          testHdfData, opt.zeroVector, opt.padding)
    end
  end
  
  if opt.nTrain < 0 then error('nTrain must be > 0') end

  -- If the user didn't specify a training set size, set it to be the
  -- training set size to be what's left after removing the validation set.
  if opt.nTrain == 0 then
    opt.nTrain = trainHdfData.y:size(1) - opt.nValidation
  end
  
  -- Verify that the validation and training sets don't overlap.
  if opt.nTrain + opt.nValidation > trainHdfData.y:size(1) then
    local expr = '(' .. opt.nTrain ' + ' .. opt.nValidation .. ' > ' .. #trainHdfData.y .. ')'
    error('nTrain + nValidation > number of training examples ' .. expr)
  end
  
  trainData = {
    labels=trainHdfData.y:narrow(1, 1, opt.nTrain),
    data=trainHdfData.X:narrow(1, 1, opt.nTrain),
    len=trainHdfData.len:narrow(1, 1, opt.nTrain),
    size=function() return opt.nTrain end
  }
  
  if opt.nValidation > 0 then
    start = trainHdfData.y:size(1) - opt.nTrain
    validData = {
       labels=trainHdfData.y:narrow(1, start, opt.nValidation),
       data=trainHdfData.X:narrow(1, start, opt.nValidation),
       len=trainHdfData.len:narrow(1, start, opt.nValidation),
       size=function() return opt.nValidation end
    }
  end
  
  if opt.test then
    testData = {
      labels=testHdfData.y,
      data=testHdfData.X,
      len=testHdfData.len,
      size=function() return testHdfData.y:size(1) end
    } 
  end

  if opt.minTrainSentLength > 0 then
    error('filtering by length is currently broken')
    local X, y = filterByLength(
        trainData.data, trainData.labels,
        opt.minTrainSentLength, 'min',
        opt.zeroVector, opt.padding)
    trainData.data = X
    trainData.labels = y
    trainData.size = function() return trainData.data:size(1) end
  elseif opt.maxTrainSentLength > 0 then
    error('filtering by length is currently broken')
    local X, y = filterByLength(
        trainData.data, trainData.labels,
        opt.maxTrainSentLength, 'max',
        opt.zeroVector, opt.padding)
    trainData.data = X
    trainData.labels = y
    trainData.size = function() return trainData.data:size(1) end
  end
  
  if opt.type == 'cuda' then
    trainData.data = trainData.data:clone():cuda()
    trainData.labels = trainData.labels:clone():cuda()
  
    if validData ~= nil then
      validData.data = validData.data:clone():cuda()
      validData.labels = validData.labels:clone():cuda()
    end
  
    if testData ~= nil then
      testData.data = testData.data:clone():cuda()
      testData.labels = testData.labels:clone():cuda()
    end
  end

  return trainData, validData, testData
end
