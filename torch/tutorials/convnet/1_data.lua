----------------------------------------------------------------------
-- This script demonstrates how to load the (SVHN) House Numbers 
-- training data, and pre-process it to facilitate learning.
--
-- The SVHN is a typicaly example of supervised training dataset. 
-- The problem to solve is a 10-class classification problem, similar
-- to the quite known MNIST challenge.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'hdf5'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'all', 'how many samples do we load: all | 20k | 1k')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
   cmd:option('-zeroVector', 107701, 'index of zero vector in dictionary: [1, dict size]')
   cmd:option('-padding', 2, 'the number of leading and trailing zero-padding entries per sentence')

   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> downloading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

-- The SVHN dataset contains 3 files:
--    + train: training data
--    + test:  test data
--    + extra: extra training data

-- By default, we don't use the extra training data, as it is much 
-- more time consuming

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

local trainHdfile = nil

if opt.size == '1k' then
  trainHdfile = hdf5.open('okanohara-train-1k.h5', 'r')
elseif opt.size == '20k' then
  trainHdfile = hdf5.open('okanohara-train-20k.h5', 'r')
else
  trainHdfile = hdf5.open('okanohara-train.h5', 'r')
end
trainDatasets = trainHdfile:read():all()

--[[
print 'X'
print(trainDatasets.X:size())
print(trainDatasets.X:select(1,1))
print 'y'
print(trainDatasets.y:size())
print(trainDatasets.y[1])
--]]

-- Load the test data.
testHdfile = hdf5.open('okanohara-test.h5', 'r')
testDatasets = testHdfile:read():all()

--[[
print 'X'
print(testDatasets.X:size())
print(testDatasets.X:select(1,1))
print 'y'
print(testDatasets.y:size())
print(testDatasets.y[1])
--]]

if opt.loss == 'mse' then
  --
  -- Input labels are 0 for negative example, 1 for positive example.
  -- Transform these to (max length-num words) for negative examples and
  -- (max length+num words) for positive examples.  This should cause the
  -- model to try to place extremely short positive and negative examples
  -- near one another and to place extremely long positive and negative
  -- examples far from one another.
  --
  local convert_to_regression_targets = function(dataset, unk_or_zero_word_index, padding)
    -- Make a binary mask of the unknown or zero-padding words in each sentence.
    --[[
    print('dataset.X')
    print(dataset.X:type())
    print(dataset.X:size())
    print('unk_or_zero_word_index')
    print(unk_or_zero_word_index)
    --]]
    unk_or_zero_mask = dataset.X:eq(unk_or_zero_word_index):int()

    -- Count the unknown and zero-padding words in each sentence by
    -- summing along the rows, then subtract the number of zero-padding
    -- words, since they don't add to the length of the sentence.
    num_unk_or_zero = torch.sum(unk_or_zero_mask, 2) - 2*padding
    num_unk_or_zero = num_unk_or_zero:reshape(num_unk_or_zero:nElement())
    num_unk_or_zero[torch.lt(num_unk_or_zero, 1)] = 1
    --[[
    print(num_unk_or_zero[torch.gt(num_unk_or_zero, 40)])
    print('num_unk_or_zero')
    print(num_unk_or_zero:size())
    --]]

    -- Make a mask of the two types of examples and use them
    -- to update the regression targets.
    neg_mask = dataset.y:eq(1)
    max_length = dataset.X:size(2) - 2*padding - 1

    -- A conditional function:
    --   target = max_length - length if negative example
    --   target = max_length + length if positive example
    reg_targets = num_unk_or_zero
    reg_targets[neg_mask] = -reg_targets[neg_mask]
    reg_targets = reg_targets + max_length

    -- Rescale to a range of about 1-10.
    reg_targets = torch.round((reg_targets:float()/8.5))+1
    return reg_targets
  end

  print '==> converting training targets for mse loss'
  trainDatasets.y = convert_to_regression_targets(
      trainDatasets, opt.zeroVector, opt.padding)
  print '==> converting test targets for mse loss'
  testDatasets.y = convert_to_regression_targets(
      testDatasets, opt.zeroVector, opt.padding)
end

trsize = trainDatasets.y:size(1)
tesize = testDatasets.y:size(1)
--[[
print 'trsize'
print(trsize)
--]]

trainData = {
   data = trainDatasets.X,
   labels = trainDatasets.y,
   size = function() return trsize end
}

testData = {
   data = testDatasets.X,
   labels = testDatasets.y,
   size = function() return tesize end
} 
