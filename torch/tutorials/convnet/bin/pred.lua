#!/usr/bin/env th

require('hdf5');
require('kttorch');
require('fbcunn');

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict using a saved model')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'path to the model')
cmd:argument('-input', 'path to test data file (HDF5 format)')
cmd:argument('-pred', 'path to which to save the predictions (HDF5 format)')
cmd:option('-record', false,
  'whether to record indices/values of activations of first convolutional layer')
cmd:text()

local opt = cmd:parse(arg or {})

local testFile = hdf5.open(opt.input, 'r')
local testData = testFile:read():all()

--[[
Load model and turn on evaluation mode.  Evaluation mode is the
counterpart of training mode.  Randomness often exists in a network in
training mode, such a random dropout of layer outputs, that result in
the same input yielding a different output.  Turning on evaluation mode
ensures that the network operates in a deterministic fashion.
--]]
local model = torch.load(opt.model)
model:evaluate()

local predOut = hdf5.open(opt.pred, 'w')

-- Get and save predictions.
local pred = nil
if opt.record then
  modules = model:findModules('nn.TemporalMaxPooling')
  assert(#modules == 1)

  indexRecorder = kttorch.IndexRecorder(modules[1])
  outputRecorder = kttorch.OutputRecorder(modules[1])
  pred = kttorch.predict(model, testData.X, { indexRecorder, outputRecorder })

  predOut:write('indices', indexRecorder.recording)
  predOut:write('output', outputRecorder.recording)
else
  pred = kttorch.predict(model, testData.X)
end

predOut:write('pred', pred)

local convLayers = model:findModules('nn.TemporalConvolution')
local filterWidth = torch.zeros(1)
filterWidth[1] = convLayers[1].kW

predOut:write('filterWidth', filterWidth)
predOut:close()
