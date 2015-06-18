require 'xlua'   -- xlua provides useful tools, like progress bars
require 'optim'  -- an optimization package, for online and batch methods

require 'cutorch'
require 'fbcunn'
require('fb.luaunit')
local torch = require('fbtorch')

function test(model, data, opts)
  local time = sys.clock()

  -- averaged param use?
  --[[
  if average then
    cachedparams = parameters:clone()
    parameters:copy(average)
  end
  --]]

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  model:evaluate()

  -- test over test data
  print('==> testing on ' .. opts.mode .. ' set:')
  for t = 1,data:size() do
    -- disp progress
    xlua.progress(t, data:size())

    -- get new sample
    local input = data.data[t]
    if opt.type == 'double' then input = input:double()
    elseif opt.type == 'cuda' then input = input:cuda() end
    local target = data.labels[t]

    -- test sample
    local pred = model:forward(input)
    if opt.loss == 'mse' then
      if pred[1] > max_class then
       pred[1] = max_class
      elseif pred[1] < min_class then
       pred[1] = min_class
      end
      pred = torch.round(pred)[1]
    end
    if opts.confusion then
      opts.confusion:add(pred, target)
    end
  end

  -- timing
  time = sys.clock() - time
  time = time / data:size()
  print("\n==> time for 1 (" .. opts.mode .. ") sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  if opts.confusion then
    print(opts.confusion)
  end

  -- update log
  if opts.logger and opts.confusion then
    opts.logger:add{['% mean class accuracy (' .. opts.mode .. ' set)'] = opts.confusion.totalValid * 100}
  end

  -- averaged param use?
  if average then
    error()
    -- restore parameters
    parameters:copy(cachedparams)
  end
  
  -- next iteration
  if opts.confusion then opts.confusion:zero() end
end
