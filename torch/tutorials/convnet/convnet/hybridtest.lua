require 'xlua'   -- xlua provides useful tools, like progress bars

function hybridTest(model, data, opts)
  local mode = opts.mode
  local dataType = opts.type
  local loss = opts.loss
  local maxClass = opts.maxClass

  local time = sys.clock()

  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  model:evaluate()

  -- test over test data
  print('==> testing on ' .. mode .. ' set:')
  for t = 1,data:size() do
    -- disp progress
    xlua.progress(t, data:size())

    -- get new sample
    local input = data.data[t]
    if dataType == 'double' then input = input:double()
    elseif dataType == 'cuda' then input = input:cuda() end
    local target = data.labels[t]

    -- test sample
    local pred = model:forward(input)
    if loss == 'mse' then
      if pred[1] > maxClass then
       pred[1] = maxClass
      elseif pred[1] < 1 then
       pred[1] = 1
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

  -- next iteration
  if opts.confusion then opts.confusion:zero() end
end
