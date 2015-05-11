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
   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
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

trainHdfile = hdf5.open('okanohara-train-20k.h5', 'r')
trainDatasets = trainHdfile:read():all()

print 'X'
print(trainDatasets.X:size())
print 'y'
print(trainDatasets.y:size())

-- trainDatasets.y = trainDatasets.y:resize(trainDatasets.y:size(1), 1)

-- trainDatasets.X = torch.Tensor(trainDatasets.X)
-- trainDatasets.y = torch.Tensor(trainDatasets.y)

trsize = trainDatasets.y:size(1)
print 'trsize'
print(trsize)

trainData = {
   data = trainDatasets.X,
   -- labels = datasets.y[1],
   labels = trainDatasets.y + 1,
   size = function() return trsize end
}

-- If extra data is used, we load the extra file, and then
-- concatenate the two training sets.

-- Finally we load the test data.
testHdfile = hdf5.open('okanohara-test.h5', 'r')
testDatasets = testHdfile:read():all()
X = testDatasets.X
y = testDatasets.y + 1

testData = {
   data = testDatasets.X,
   -- labels = datasets.y[1],
   labels = testDatasets.y,
   size = function() return testDatasets.y:size(1) end
} 

print '==> data'
print(trainData.data:size())
print '==> labels'
print(trainData.labels:size())
print '==> trsize'
print(trainData:size())

-- The example data is loaded and transformed into two objects:
-- ==> data    
-- 
--  73257
--      3
--     32
--     32
-- [torch.LongStorage of size 4]
-- 
-- ==> labels  
-- 
--  73257
-- [torch.LongStorage of size 1]
-- 
-- Since I'll be using a lookup table, I need to transform the Okanohara data into something like:
-- 
--    250k
--      45
-- [torch.LongStorage of size 2]
-- 
-- The label can remain a [torch.LongStorage of size 1].

----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

-- trainData.data = trainData.data:float()
-- testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.
