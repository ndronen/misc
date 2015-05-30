require 'torch'   -- torch
require 'image'   -- for color transforms
require 'hdf5'

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
   cmd:option('-scaleMseTarget', false, 'whether to scale the target variable when loss is mse')
   cmd:option('-zeroVector', 107701, 'index of zero vector in dictionary: [1, dict size]')
   cmd:option('-padding', 2, 'the number of leading and trailing zero-padding entries per sentence')
   cmd:option('-type', 'double', 'type: double | float | cuda')
   cmd:text()
   opt = cmd:parse(arg or {})
end

local trainHdfile = nil

if opt.size == '1k' then
  trainHdfile = hdf5.open('okanohara-train-1k.h5', 'r')
elseif opt.size == '20k' then
  trainHdfile = hdf5.open('okanohara-train-20k.h5', 'r')
else
  trainHdfile = hdf5.open('okanohara-train.h5', 'r')
end
trainDatasets = trainHdfile:read():all()

-- Load the test data.
testHdfile = hdf5.open('okanohara-test.h5', 'r')
testDatasets = testHdfile:read():all()

if (opt.loss == 'mse') and (opt.scaleMseTarget) then
  --[[
  Input labels are 0 for negative example, 1 for positive example.
  Transform these to (max length-num words) for negative examples and
  (max length+num words) for positive examples.  This should cause the
  model to try to place extremely short positive and negative examples
  near one another and to place extremely long positive and negative
  examples far from one another.
  --]]
  local convert_to_regression_targets = function(dataset, unk_or_zero_word_index, padding)
    -- Make a binary mask of the unknown or zero-padding words in each sentence.
    unk_or_zero_mask = dataset.X:eq(unk_or_zero_word_index):int()

    --[[
    Count the unknown and zero-padding words in each sentence by summing
    along the rows, then subtract the number of zero-padding words,
    since they don't add to the length of the sentence.
    --]]
    num_unk_or_zero = torch.sum(unk_or_zero_mask, 2) - 2*padding
    num_unk_or_zero = num_unk_or_zero:reshape(num_unk_or_zero:nElement())
    num_unk_or_zero[torch.lt(num_unk_or_zero, 1)] = 1

    --[[
    Make a mask of the two types of examples and use them to update the
    regression targets.
    --]]
    neg_mask = dataset.y:eq(1)
    max_length = dataset.X:size(2) - 2*padding - 1

    --[[
    A conditional function:
      target = max_length - length if negative example
      target = max_length + length if positive example
    --]]
    reg_targets = num_unk_or_zero
    reg_targets[neg_mask] = -reg_targets[neg_mask]
    reg_targets = reg_targets + max_length

    -- Rescale to a range of about 1-10.
    reg_targets = torch.round((reg_targets:float()/8.5))+1
    return reg_targets
  end

  trainDatasets.y = convert_to_regression_targets(
      trainDatasets, opt.zeroVector, opt.padding)
  testDatasets.y = convert_to_regression_targets(
      testDatasets, opt.zeroVector, opt.padding)
end

trsize = trainDatasets.y:size(1)
tesize = testDatasets.y:size(1)

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

if opt.type == 'cuda' then
  trainData.data = trainData.data:clone():cuda()
  trainData.labels = trainData.labels:clone():cuda()
  testData.data = testData.data:clone():cuda()
  testData.labels = testData.labels:clone():cuda()
end
