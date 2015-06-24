require 'xlua'    -- xlua provides useful tools, like progress bars

function train(model, trainData, args)
  local optimState = args.optimState
  local optimMethod = args.optimMethod
  local parameters = args.parameters
  local gradParameters = args.gradParameters
  local confusion = args.confusion
  local epoch = args.epoch
  local trainLogger = args.trainLogger
  local normsLogger = args.normsLogger
  local batchSize = args.batchSize
  local dataType = args.type
  local loss = args.loss
  local spatial = args.spatial
  local save = args.save
  local renormFreq = args.renormFreq
  local zeroVector = args.zeroVector
  local zeroZeroVector = args.zeroZeroVector
  local maxWordNorm = args.maxWordNorm

  -- local vars
  local time = sys.clock()

  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training()

  -- shuffle at each epoch
  local shuffle = torch.randperm(trainData.size())

  -- do one epoch
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

  local iter = 0
  for t = 1,trainData:size(),batchSize do
    iter = iter + 1

    -- disp progress
    xlua.progress(t, trainData:size())

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+batchSize-1,trainData:size()) do
      -- load new sample
      local input = trainData.data[shuffle[i]]
      local target = trainData.labels[shuffle[i]]
      if dataType == 'double' then input = input:double()
      elseif dataType == 'cuda' then input = input:cuda() end
      table.insert(inputs, input)
      table.insert(targets, target)
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- f is the average of all criterions
      local f = 0

      -- evaluate function for complete mini batch
      for i = 1,#inputs do
        -- input = inputs[i]:clone():cuda()
        local input = inputs[i]

        local output = model:forward(input)
        local targs = targets[i]

        if loss == 'mse' then
          if dataType == 'cuda' then
            targs = torch.CudaTensor(1)
          else
            targs = torch.Tensor(1)
          end
          targs[1] = targets[i]
        else
          if spatial then
            if dataType == 'cuda' then
              targs = torch.CudaTensor(1)
            else
              targs = torch.Tensor(1)
            end
            targs[1] = targets[i]
          else
            targs = targets[i]
          end
        end

        local err = criterion:forward(output, targs)
        f = f + err

        -- estimate df/dW
        local df_do = criterion:backward(output, targs)
        model:backward(input, df_do)

        if loss == 'mse' then
          if output[1] > max_class then
            output[1] = max_class
          elseif output[1] < 1 then
            output[1] = 1
          end
          output = torch.round(output)[1]
        end

        if torch.isTensor(targs) then
          confusion:add(output, targs[1])
        else
          confusion:add(output, targs)
        end
      end

      -- normalize gradients and f(X)
      gradParameters:div(#inputs)
      f = f/#inputs

      -- return f and df/dX
      return f, gradParameters
    end

    -- Optimize on current mini-batch.
    optimMethod(feval, parameters, optimState)

    -- If there's a zero vector, ensure that it's always 0.
    if zeroZeroVector then
      for i,module in ipairs(model:listModules()) do
        if torch.isTypeOf(module, 'nn.LookupTable') then
          module.weight[zeroVector]:zero()
        end
      end
    end

    p = 2
    renormDim = 1

    if (renormFreq > 0) and (iter % renormFreq == 0) then
      -- Rescale weights of fully-connected and convolutional layers.
      renormer:renorm()
  
      -- Rescale word representations.  I can't use Renormer to do this yet.
      for i,module in ipairs(model:listModules()) do
        if torch.isTypeOf(module, 'nn.LookupTable') then
          -- The following commented-out line of code is what I should be
          -- able to run, but it causes an error that I haven't been able
          -- to track down.  The uncommented-out line is the workaround.
          -- module.weight:renorm(p, renormDim, opt.maxWordNorm)
          weight = module.weight:clone():float():renorm(p, renormDim, opt.maxWordNorm)
          if dataType == 'cuda' then
            weight = weight:cuda()
          end
          module.weight = weight
        end
      end
    end
  end

  -- time taken
  time = sys.clock() - time
  time = time / trainData:size()
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)

  -- update logger
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

  -- TODO: log summary statistic of model weights.

  -- save/log current net
  local filename = paths.concat(save, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  torch.save(filename, model)

  for i,module in ipairs(model:listModules()) do
    if module.weight ~= nil then
      norms = module.weight:norm(2, 2)
      normsLogger:add({
        string.format('%d', epoch),
        string.format('%d', i),
        torch.typename(module),
        string.format('%f', torch.min(norms)),
        string.format('%f', torch.max(norms)),
        string.format('%f', torch.mean(norms))})
    end
  end
    
  -- next epoch
  confusion:zero()
end
