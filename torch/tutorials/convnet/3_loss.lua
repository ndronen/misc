require 'cutorch'
require 'fbcunn'
require('fb.luaunit')
local torch = require('fbtorch')

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Grammaticality model loss')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
   cmd:option('-type', 'double', 'type: double | float | cuda')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

print '==> define loss'

if opt.loss == 'margin' then
  nclasses = 2
  -- This loss takes a vector of classes, and the index of
  -- the grountruth class as arguments. It is an SVM-like loss
  -- with a default margin of 1.
  criterion = nn.MultiMarginCriterion()
elseif opt.loss == 'nll' then
  nclasses = 2
  -- This loss takes a vector of classes, and the index of
  -- This loss requires the outputs of the trainable model to
  -- be properly normalized log-probabilities, which can be
  -- achieved using a softmax function

  model:add(nn.LogSoftMax())
  -- The loss works like the MultiMarginCriterion: it takes
  -- a vector of classes, and the index of the grountruth class
  -- as arguments.

  criterion = nn.ClassNLLCriterion()
elseif opt.loss == 'mse' then
  -- for MSE, we add a tanh, to restrict the model's output
  -- model:add(nn.Tanh())

  -- The mean-square error is not recommended for classification
  -- tasks, as it typically tries to do too much, by exactly modeling
  -- the 1-of-N distribution. For the sake of showing more examples,
  -- we still provide it here:

  criterion = nn.MSECriterion()
  criterion.sizeAverage = false

  -- Compared to the other losses, the MSE criterion needs a distribution
  -- as a target, instead of an index. Indeed, it is a regression loss!
  -- So we need to transform the entire label vectors:
end

print(criterion)
