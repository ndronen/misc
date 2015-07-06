#!/usr/bin/env th

require('hdf5');
require('kttorch');
require('fbcunn');
require('convnet.inspect');
require('convnet.utils');

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify filters of a convnet')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'path to a serialized model')
cmd:argument('-data', 'path to data for testing (HDF5 format with data in "X" and labels in "y")')
cmd:argument('-output', 'path to file to which to write output')
cmd:text()

local opt = cmd:parse(arg or {})

-- Load data and model.
local dataFile = hdf5.open(opt.data, 'r')
local data = dataFile:read():all()
local model = loadModel(opt.model)

-- Run the data through the model while recording the activations of the
-- filters, then determine whether filters are conditioned to detect
-- features of positive or negative examples.
local recordingOpts = { activations=true }
local pred, recording = predictAndRecordConvolution(model, data.X, recordingOpts)
local filterInfo = computeFilterPolarities(data.y, recording.activations)

-- Add filter width to output.
local modules = model:findModules('nn.TemporalConvolution')
assert(#modules == 1)
filterInfo.filterWidth = torch.Tensor(1):fill(modules[1].kW)

local output = hdf5.open(opt.output, 'w')

for k,v in pairs(filterInfo) do
  output:write(k, v)
end

output:close()
