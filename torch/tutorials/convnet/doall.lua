#!/usr/bin/env th

require 'cutorch'
require 'fbcunn'

require('fb.luaunit')
local torch = require('fbtorch')

torch.setnumthreads(6)
torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a CNN')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
cmd:option('-scaleMseTarget', 0, 'whether to scale the target variable when loss is mse')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | ADAGRAD | ADADELTA')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-zeroVector', 107701, 'index of zero vector in dictionary: [1, dict size]')
cmd:option('-padding', 2, 'the number of leading and trailing zero-padding entries per sentence')
cmd:option('-size', 'all', 'how many samples do we load for training: all | smakll')
cmd:option('-kernelWidth', 2, 'width of kernels: 2 or greater')
cmd:option('-nKernels', 500, 'number of kernels: 2 or greater')
cmd:option('-nFullyConnectedLayers', '1', 'number of extra fully-connected layers after convolutional layers')
cmd:option('-lookupOnGpu', true, 'put the lookup table on the GPU')
cmd:option('-maxNorm', 1, 'maximum 2-norm of neuron weights in fully-connected layers')
cmd:option('-maxWordNorm', 1, 'maximum 2-norm of word representations in lookup table')
cmd:option('-wordDims', 50, 'number of dimensions of word representations')
cmd:option('-word2Vec', false, 'use pretrained word2vec weights in lookup table')
cmd:option('-fixWords', false, 'disable updates of word representations')
cmd:option('-nValidation', 0, 'size of the validation set to hold out from training')
cmd:option('-test', false, 'whether to load and predict on test set')
cmd:option('-activation', 'relu', 'activation function: relu | tanh')
cmd:option('-renormFreq', 5, 'number of updates after which to renorm weights')
cmd:option('-spatial', false, 'train a spatial convolutional network')
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'float' then
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

print('1_data.lua')
dofile '1_data.lua'
print('2_model.lua')
dofile '2_model.lua'
print('3_loss.lua')
dofile '3_loss.lua'
print('4_train.lua')
dofile '4_train.lua'
print('5_test.lua')
dofile '5_test.lua'

print(model)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
validationLogger = optim.Logger(paths.concat(opt.save, 'validation.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
normsLogger = optim.Logger(paths.concat(opt.save, 'norms.log'))
normsLogger:setNames({'epoch', 'layer#', 'module', 'min', 'max', 'mean'})

while true do
  train()
  -- TODO: change test() to return error; save the model here if validation
  -- performance is best one yet.
  if validData ~= nil then
    test(validData, { mode='validation', logger=validationLogger })
  end
  if testData ~= nil then
    test(testData, { mode='test', logger=testLogger })
  end
end
