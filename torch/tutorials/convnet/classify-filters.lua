#!/usr/bin/env th

require('hdf5');
require('kttorch');
require('fbcunn');

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify filters of a convnet')
cmd:text()
cmd:text('Options:')
cmd:argument('-input', 'path to test data file (HDF5 format)')
cmd:argument('-pred', 'path to prediction file (HDF5 format)')
cmd:argument('-analysis', 'path to analysis file (HDF5 format)')
cmd:text()

local opt = cmd:parse(arg or {})

-- Load test data and predictions.
local testFile = hdf5.open(opt.input, 'r')
local testData = testFile:read():all()

-- Prediction data has fields indices, output, and pred.
local predFile = hdf5.open(opt.pred, 'r')
local predData = predFile:read():all()

local anOut = hdf5.open(opt.analysis, 'w')

--[[
Analysis consists of a classification of the filters of a convnet.
1) Compute mean output of each filter for the positive examples.
2) Compute mean output of each filter for the negative examples.
3) Take differences d of those means (mean_pos - mean_neg).
4) Positive filters are those where d > 0, negative d < 0.
--]]

local POSITIVE = 2
local NEGATIVE = 1

ncol = predData.output:size(2)

posIndices = testData.y:eq(POSITIVE)
posIndices:resize(posIndices:size(1), 1)
posIndices = posIndices:expand(predData.output:size(1), predData.output:size(2))
posOutput = predData.output[posIndices]
posOutput:resize(posOutput:nElement()/ncol, ncol)
meanPos = posOutput:mean(1)

negIndices = testData.y:eq(NEGATIVE)
negIndices:resize(negIndices:size(1), 1)
negIndices = negIndices:expand(predData.output:size(1), predData.output:size(2))
negOutput = predData.output[negIndices]
negOutput:resize(negOutput:nElement()/ncol, ncol)
meanNeg = negOutput:mean(1)

diffMean = meanPos - meanNeg
posFilters = diffMean:gt(0)
negFilters = diffMean:lt(0)
disabledFilters = diffMean:eq(0)

anOut:write('meanPos', meanPos)
anOut:write('meanNeg', meanNeg)
anOut:write('diffMean', diffMean)
anOut:write('posFilters', posFilters)
anOut:write('negFilters', negFilters)
anOut:write('disabledFilters', disabledFilters)

anOut:close()
