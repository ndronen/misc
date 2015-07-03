require('kttorch');

predictAndRecordConvolution = function(model, data, opts) 
  local opts = opts or {}
  local recordIndices = opts.indices or false
  local recordActivations = opts.activations or false
  local recorders = {}

  if recordIndices or recordActivations then
    local modules = model:findModules('nn.TemporalMaxPooling')
    if #modules == 0 then
      modules = model:findModules('nn.TemporalKMaxPooling')
    end

    if recordIndices then
      recorders.indices = kttorch.IndexRecorder(modules[1])
    end
    if recordActivations then
      recorders.activations = kttorch.OutputRecorder(modules[1])
    end
  end

  local pred = kttorch.predict(model, data, recorders)
  local recordings = {}
  if recordIndices then
    recordings.indices = recorders.indices:getRecording()
  end
  if recordActivations then
    recordings.activations = recorders.activations:getRecording()
  end
  
  return pred, recordings
end

computeFilterPolarities = function(labels, activations, opts)
  --[[
  1) Compute mean output of each filter for the positive examples.
  2) Compute mean output of each filter for the negative examples.
  3) Take differences d of those means (mean_pos - mean_neg).
  4) Positive filters are those where d > 0, negative d < 0.
  --]]
  
  local opts = opts or {}
  local negativeLabel = opts.negativeLabel or 1
  local positiveLabel = opts.positiveLabel or 2
  
  local ncol = activations:size(2)
  
  local posIndices = labels:eq(positiveLabel)
  posIndices:resize(posIndices:size(1), 1)
  posIndices = posIndices:expand(posIndices:size(1), activations:size(2))
  local posOutput = activations[posIndices]
  posOutput:resize(posOutput:nElement()/ncol, ncol)
  local meanPos = posOutput:mean(1)
  
  local negIndices = labels:eq(negativeLabel)
  negIndices:resize(negIndices:size(1), 1)
  negIndices = negIndices:expand(negIndices:size(1), activations:size(2))
  local negOutput = activations[negIndices]
  negOutput:resize(negOutput:nElement()/ncol, ncol)
  local meanNeg = negOutput:mean(1)
  
  local polarities = meanPos - meanNeg
  local posFilters = polarities:gt(0)
  local negFilters = polarities:lt(0)
  local disabledFilters = polarities:eq(0)

  return {
    positiveFilters=posFilters,
    negativeFilters=negFilters,
    disabledFilters=disabledFilters,
    polarities=polarities
  }
end

countPolarityOfActivations = function(model, input, positiveFilters, negativeFilters, filterWidth)
  -- Assuming input is a single vector for now.
  local recordingOpts = { activations=true, indices=true }
  local input = input:clone():resize(1, input:size(1))
  local pred, recording = predictAndRecordConvolution(model, input, recordingOpts)

  local posPolarity = torch.zeros(input:size(2))
  local negPolarity = torch.zeros(input:size(2))

  for i=1,torch.max(recording.indices) do
    posPolarity[i] = torch.sum(recording.indices:eq(i):cmul(filterInfo.positiveFilters))
    negPolarity[i] = torch.sum(recording.indices:eq(i):cmul(filterInfo.negativeFilters))
  end

  return posPolarity, negPolarity
end

sumMagnitudeOfActivations = function(model, input, positiveFilters, negativeFilters, filterWidth)

end
