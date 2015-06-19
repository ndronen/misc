buildCriterion = function(model, opt)
  if opt.loss == 'margin' then
    nclasses = 2
    criterion = nn.MultiMarginCriterion()
  elseif opt.loss == 'nll' then
    nclasses = 2
    model:add(nn.LogSoftMax())
    criterion = nn.ClassNLLCriterion()
  elseif opt.loss == 'mse' then
    criterion = nn.MSECriterion()
    criterion.sizeAverage = false
  end
  return criterion
end
