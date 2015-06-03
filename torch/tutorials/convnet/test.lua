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
cmd:argument('-output', 'path to which to save the predictions')
cmd:text()

local opt = cmd:parse(arg or {})

print('opt.model ' .. opt.model)
print('opt.input ' .. opt.input)
print('opt.output ' .. opt.output)

-- Load data and model.
local testFile = hdf5.open(opt.input, 'r')
local testData = testFile:read():all()
local model = torch.load(opt.model)

-- Get and save predictions.
local pred = kttorch.predict(model, testData.X)
local predOut = hdf5.open(opt.output, 'w')
predOut:write('pred', pred)
predOut:close()

