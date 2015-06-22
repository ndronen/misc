require 'hdf5'

--[[
--]]
makeCollobertAndWestonNegativeExamples = function(data, labels, opts)
  data = data:clone()
  labels = labels:clone()

  local opts = opts or {}

  local negativeLabel = opts.negativeLabel
  if negativeLabel == nil then
    negativeLabel = 1
  end

  local maxIndex = opts.maxIndex
  if maxIndex == nil then
    maxIndex = torch.max(data)
  end

  local padding = opts.padding
  if padding == nil then
    padding = 2
  end

  -- Find examples that have the negative class's label and replace
  -- one word with a random word.  A word here is an index into the
  -- vocabulary.
  for i=1,data:size(1) do
    if labels[i] == negativeLabel then
      newWord = torch.random(maxIndex)
      replacementIndex = torch.random(padding + 1, data:size(2) - padding)
      data[i][replacementIndex] = newWord
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
  if not file_exists(opt.trainFile) then
    error('Train data file does not exist ' .. opt.trainFile)
  end
  
  local trainHdfFile = hdf5.open(opt.trainFile, 'r')
  local trainHdfData = trainHdfFile:read():all()
  
  local testHdfFile = nil
  local testHdfData = nil
  if opt.test then
    if not file_exists(opt.testFile) then
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
    size=function() return opt.nTrain end
  }
  
  if opt.nValidation > 0 then
    start = trainHdfData.y:size(1) - opt.nTrain
    validData = {
       labels=trainHdfData.y:narrow(1, start, opt.nValidation),
       data=trainHdfData.X:narrow(1, start, opt.nValidation),
       size=function() return opt.nValidation end
    }
  end
  
  if opt.test then
    testData = {
      labels = testHdfData.y,
      data = testHdfData.X,
      size = function() return testHdfData.y:size(1) end
    } 
  end

  if opt.minTrainSentLength > 0 then
    local X, y = filterByLength(
        trainData.data, trainData.labels,
        opt.minTrainSentLength, 'min',
        opt.zeroVector, opt.padding)
    trainData.data = X
    trainData.labels = y
    trainData.size = function() return trainData.data:size(1) end
  elseif opt.maxTrainSentLength > 0 then
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
