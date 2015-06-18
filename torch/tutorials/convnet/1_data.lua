require 'torch'   -- torch
require 'image'   -- for color transforms
require 'hdf5'

dofile('utils.lua')

--[[
Input labels are 0 for negative example, 1 for positive example.
Transform these to (max length-num words) for negative examples and (max
length+num words) for positive examples.  This should cause the model
to try to place extremely short positive and negative examples near one
another and to place extremely long positive and negative examples far
from one another.
--]]
local scaleRegressionTargets = function(dataset, zeroVectorIndex, padding)
  -- Make a binary mask of the unknown or zero-padding words in each sentence.
  zeroVectorMask = dataset.X:eq(zeroVectorIndex):int()

  --[[
  Count the unknown and zero-padding words in each sentence by summing
  along the rows, then subtract the number of zero-padding words,
  since they don't add to the length of the sentence.
  --]]
  numZero = torch.sum(zeroVectorMask, 2) - 2*padding
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

if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Grammaticality model data set loading')
   cmd:text()
   cmd:text('Arguments:')
   cmd:argument('-trainFile', 'HDF5 file containing training data')
   cmd:text('Options:')
   cmd:option('-test', false, "whether to predict on test data")
   cmd:option('-testFile', "nil", 'HDF5 file containing test data')
   cmd:option('-nTrain', 0, 'size of the training set (taken from first nTrain elements of training set)')
   cmd:option('-nValidation', 0, 'size of the validation set to hold out from training (taken from last nValidation elements of training set)')
   cmd:option('-type', 'double', 'type: double | float | cuda')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
   cmd:option('-scaleMseTarget', 0, 'whether to scale the target variable when loss is mse')
   cmd:option('-zeroVector', 107701, 'index of zero vector in dictionary: [1, dict size]')
   cmd:option('-padding', 2, 'the number of leading and trailing zero-padding entries per sentence')
   cmd:text()
   opt = cmd:parse(arg or {})
end

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

print('nTrain ' .. opt.nTrain)

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
