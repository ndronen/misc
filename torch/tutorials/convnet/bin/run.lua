#!/usr/bin/env th

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Grammaticality model')
cmd:text()

cmd:text('Arguments:')
cmd:argument('nWords', 'number of words in the lookup table', 'string')
cmd:argument('trainFile', 'HDF5 file containing training data', 'string')

cmd:text('Options:')
cmd:option('-test', false, 'whether to load and predict on test set')
cmd:option('-testFile', "nil", 'HDF5 file containing test data')
cmd:option('-nTrain', 0, 'size of the training set (taken from first nTrain elements of training set)')
cmd:option('-nValidation', 0, 'size of the validation set to hold out from training (taken from last nValidation elements of training set)')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
cmd:option('-scaleMseTarget', 0, 'whether to scale the target variable when loss is mse')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | ADAGRAD | ADADELTA')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-zeroVector', 0, 'index of zero vector in dictionary: [1, dict size]; 0 means there is no zero vector')
cmd:option('-zeroZeroVector', false, 'always undo any weight updates to the unknown word zero vector')
cmd:option('-lookupTableDropout', 0, 'apply dropout to input before lookup table layer')
cmd:option('-padding', 2, 'the number of leading and trailing zero-padding entries per sentence')
cmd:option('-kernelWidth', 2, 'width of kernels: 2 or greater')
cmd:option('-nKernels', 500, 'number of kernels: 2 or greater')
cmd:option('-nFullyConnectedLayers', '1', 'number of extra fully-connected layers after convolutional layers')
cmd:option('-maxNorm', 1, 'maximum 2-norm of neuron weights in fully-connected layers')
cmd:option('-maxWordNorm', 0, 'maximum 2-norm of word representations in lookup table')
cmd:option('-forceWordsOnBall', false, 'keep 2-norm of word representations on same surface (requires -maxWordNorm > 0)')
cmd:option('-maxFilterNorm', 0, 'maximum 2-norm of components of convolutional filters')
cmd:option('-forceFiltersOnBall', false, 'keep 2-norm of components of convolutional filters on same surface (requires -maxFilterNorm > 0)')
cmd:option('-wordDims', 50, 'number of dimensions of word representations')
cmd:option('-word2Vec', false, 'use pretrained word2vec weights in lookup table')
cmd:option('-fixWords', false, 'disable updates of word representations')
cmd:option('-activation', 'relu', 'activation function: relu | tanh')
cmd:option('-renormFreq', 0, 'number of updates after which to renorm weights')
cmd:option('-spatial', false, 'train a spatial convolutional network')
cmd:option('-minTrainSentLength', 0, 'minimum length of a sentence for training: (disabled=0)')
cmd:option('-maxTrainSentLength', 0, 'maximum length of a sentence for training: (disabled=0)')
cmd:option('-makeCollobertNegativeExamples', false, 'whether to make Collobert & Weston-style negative examples; when true, negative valdiation and test set examples are made ones and negative training set examples are made anew before each epoch')
cmd:option('-makePermutationNegativeExamples', false, 'whether to make permutation-style negative examples; when true, negative valdiation and test set examples are made ones and negative training set examples are made anew before each epoch')

cmd:text()

local opt = cmd:parse(arg or {})

require 'convnet.utils'
require 'convnet.data'
require 'convnet.model'
require 'convnet.loss'
require 'convnet.optim'
require 'convnet.train'
require 'convnet.test'

if opt.type == 'float' then
  require 'nn'
  torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
  -- require 'cunn'
  require 'fbcunn'
  torch = require('fbtorch')
  torch.setdefaulttensortype('torch.FloatTensor')
end

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

if opt.zeroZeroVector then
  if opt.zeroVector == 0 then
    error('Use -zeroVector INDEX with -zeroZeroVector')
  end
end

local trainData, validData, testData = loadData(opt)
local model, renormers = buildModel(opt)
local criterion = buildCriterion(model, opt)

if opt.type == 'cuda' then
  model:cuda()
  criterion:cuda()
end

print(model)

local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local validationLogger = optim.Logger(paths.concat(opt.save, 'validation.log'))
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
local normsLogger = optim.Logger(paths.concat(opt.save, 'norms.log'))
normsLogger:setNames({'epoch', 'layer#', 'module', 'min', 'max', 'mean'})

local optimState, optimMethod = buildOptimizer(opt)
local confusion = buildConfusionMatrix(trainData.labels)
local parameters, gradParameters = model:getParameters()
local epoch = 1

local negativeExampleMaker = nil

if opt.makeCollobertNegativeExamples and opt.makePermutationNegativeExamples then
  error('Collobert and permutation negative examples are mutually exclusive')
end

if opt.makeCollobertNegativeExamples then
  negativeExampleMaker = makeCollobertAndWestonNegativeExamples
elseif opt.makePermutationNegativeExamples then
  negativeExampleMaker = makePermutationNegativeExamples
end

if negativeExampleMaker then
  if validData ~= nil then
    local data, labels = negativeExampleMaker(
        validData.data, validData.labels, { lengths=validData.len })
    validData.data = data
    validData.labels = labels
  end
  if testData ~= nil then
    local data, labels = negativeExampleMaker(
        testData.data, testData.labels, { lengths=testData.len })
    testData.data = data
    testData.labels = labels
  end
end

while true do
  local trainDataForEpoch = {
    data=trainData.data,
    labels=trainData.labels,
    size=trainData.size
  }

  if opt.makeNegativeExamples then
    local data, labels = negativeExampleMaker(
        trainData.data, trainData.labels, { lengths=trainData.len })
    trainDataForEpoch.data = data
    trainDataForEpoch.labels = labels
  end

  -- TODO: change train() to return predictions.
  train(model, trainDataForEpoch, {
    optimState=optimState,
    optimMethod=optimMethod,
    parameters=parameters,
    gradParameters=gradParameters,
    confusion=confusion,
    epoch=epoch,
    trainLogger=trainLogger,
    normsLogger=normsLogger,
    batchSize=opt.batchSize,
    dataType=opt.type,
    loss=opt.loss,
    spatial=opt.spatial,
    save=opt.save,
    renormFreq=opt.renormFreq,
    zeroVector=opt.zeroVector,
    zeroZeroVector=opt.zeroZeroVector,
    renormers=renormers
  })

  -- TODO: change test() to return predictions so we don't need to pass in the
  -- logger or confusion matrix objects.
  -- TODO: save the model here if validation performance is best one yet.
  if validData ~= nil then
    test(model, validData, { mode='validation', logger=validationLogger, confusion=confusion, loss=opt.loss, type=opt.type })
  end
  if testData ~= nil then
    test(model, testData, { mode='test', logger=testLogger, confusion=confusion, loss=opt.loss, type=opt.type })
  end

  epoch = epoch + 1
end
