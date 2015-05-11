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

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 2-class problem
noutputs = 2

-- input dimensions
-- nfeats = 3
-- width = 32
-- height = 32
-- ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
-- nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
-- nstates = {64,64,128}
-- filtsize = 5
-- poolsize = 2
normkernel = image.gaussian1D(7)

print '==> sentence length'
print(trainData.data:size(2))

nWords = 107701
nWordDims = 50
inputFrameSize = nWordDims; -- dimensionality of one sequence element 
outputFrameSize = 13;       -- number of derived features for one sequence element
kw = 3;          -- kernel spans three input elements
dw = 1;          -- we step once and go on to the next sequence element

----------------------------------------------------------------------
print '==> construct model'

if opt.type == 'cuda' then
  -- a typical modern convolution network (conv+relu+pool)
  model = nn.Sequential()

  -- stage 1: lookup table
  model:add(nn.LookupTable(nWords, nWordDims))

  -- stage 2: filter bank -> squashing -> pooling

  model:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kw, dw))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(trainData.data:size(2)-2))

  -- stage 3: linear output
  model:add(nn.Linear(outputFrameSize, noutputs))

  -- stage 3 : standard 2-layer neural network
  -- model:add(nn.View(nstates[2]*filtsize*filtsize))
  -- model:add(nn.Dropout(0.5))
  -- model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
  -- model:add(nn.ReLU())
  -- model:add(nn.Linear(nstates[3], noutputs))

else
  -- a typical convolutional network, with locally-normalized hidden
  -- units, and L2-pooling

  -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
  -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
  -- the use of LP-pooling (with P=2) has a very positive impact on
  -- generalization. Normalization is not done exactly as proposed in
  -- the paper, and low-level (first layer) features are not fed to
  -- the classifier.

  model = nn.Sequential()

  -- stage 1: lookup table
  model:add(nn.LookupTable(nWords, nWordDims))

  print '==> nn.TemporalConvolution'
  print 'inputFrameSize'
  print(inputFrameSize)
  print 'outputFrameSize'
  print(outputFrameSize)
  print 'kw'
  print(kw)
  print 'dw'
  print(dw)

  -- stage 2: filter bank -> squashing -> pooling
  model:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kw, dw))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(trainData.data:size(2)-2))

  -- stage 3: linear output
  model:add(nn.View(outputFrameSize))
  model:add(nn.Linear(outputFrameSize, noutputs))

  -- stage 3 : standard 2-layer neural network
  -- model:add(nn.View(nstates[2]*filtsize*filtsize))
  -- model:add(nn.Dropout(0.5))
  -- model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
  -- model:add(nn.ReLU())
  -- model:add(nn.Linear(nstates[3], noutputs))
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
