require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

require 'cutorch'
require 'fbcunn'
require('fb.luaunit')
local torch = require('fbtorch')

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | ADAGRAD | ADADELTA')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-maxNorm', 10, 'maximum 2-norm of neuron weights in fully-connected layers') 
   cmd:option('-maxWordNorm', 20, 'maximum 2-norm of word representations in lookup table')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-renormFreq', 0, 'number of updates after which to renorm weights')
   cmd:option('-zeroVector', 107701, 'index of zero vector in dictionary: [1, dict size]')
   cmd:text()
   opt = cmd:parse(arg or {})
end

print '==> setting up classes'
local classes = {}
min_class = 1
max_class = torch.max(trainData.labels)
print('==> setting up classes ' .. min_class .. ' ' .. max_class)
for i=1,max_class do 
  table.insert(classes, tostring(i))
end

if opt.type == 'cuda' then
  model:cuda()
  criterion:cuda()
end

print '==> creating confusion matrix'
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(max_class)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the model
-- into a 1-dim vector

print '==> getting initial parameters'
if model then
   parameters, gradParameters = model:getParameters()
end

if opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd
elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd
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

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   local iter = 0
   for t = 1,trainData:size(),opt.batchSize do
     iter = iter + 1

      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
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
                          local targs = nil
                          if opt.loss == 'mse' then
                            if opt.type == 'cuda' then
                              targs = torch.CudaTensor(1)
                            else
                              targs = torch.Tensor(1)
                            end
                            targs[1] = targets[i]
                          else
                            targs = targets[i]
                          end

                          local err = criterion:forward(output, targs)
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targs)
                          model:backward(input, df_do)

                          if opt.loss == 'mse' then
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
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end

      -- If there's a zero vector, ensure that it's always 0.
      if opt.zeroVector and opt.zeroVector ~= 0 then
        for i,module in ipairs(model:listModules()) do
          if torch.isTypeOf(module, 'nn.LookupTable') then
            module.weight[opt.zeroVector]:zero()
          end
        end
      end

     p = 2
     renormDim = 1

     if (opt.renormFreq > 0) and (iter % opt.renormFreq == 0) then
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
           if opt.type == 'cuda' then
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

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

   -- TODO: log summary statistic of model weights.

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
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
   epoch = epoch + 1
end
