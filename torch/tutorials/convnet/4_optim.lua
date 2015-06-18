require 'optim'   -- an optimization package, for online and batch methods

buildOptimizer = function(opt)
  local optimState = nil
  local optimMethod = nil

  if opt.optimization == 'SGD' then
    optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
    }
    optimMethod = optim.sgd
  elseif opt.optimization == 'ADAGRAD' then
    optimState = {
      learningRate = opt.learningRate
    }
    optimMethod = optim.adagrad
  elseif opt.optimization == 'ADADELTA' then
    optimState = nil
    optimMethod = optim.adadelta
  else
    error('unknown optimization method')
  end

  return optimState, optimMethod
end
