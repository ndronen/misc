require 'torch';
require 'convnet.data'

---------------------------------------------------------
-- HELPER FUNCTIONS
---------------------------------------------------------

verifyNumberOfElementsModified = function(data, newData, n)
  local nDiff = 0

  for i=1,data:size(2) do
    if newData[1][i] ~= data[1][i] then
      nDiff = nDiff + 1
    end
  end

  if nDiff ~= n then
    print("incorrect number of elements modified.  expected " .. tostring(n) ..
      " got " .. tostring(nDiff))
    print(data)
    print(newData)
  end

  return nDiff == n
end

verifyModifiedElementsAreContiguous = function(data, newData, n)
  local prevDiff = nil
  local continguous = true

  for i=1,data:size(2) do
    if newData[1][i] ~= data[1][i] then
      if i > 1 and prevDiff ~= nil then
        if i - prevDiff > 1 then
          contiguous = false
        end
      end
      prevDiff = i
    end
  end
  
  return continguous == true
end

verifyPaddingIsUnchanged = function(data, newData, padding)
  local unchanged = true

  for i=1,padding do
    if newData[1][i] ~= 0 then
      unchanged = false
    end
  end

  for i=data:size(2),data:size(2)+1-padding,-1 do
    if newData[1][i] ~= 0 then
      unchanged = false
    end
  end

  return unchanged == true
end

---------------------------------------------------------
-- TESTS
---------------------------------------------------------

testNoiseCountPermutation = function()
  local data = torch.Tensor({ { 0, 0, 1, 2, 3, 4, 5, 0, 0 } })
  local labels = torch.Tensor({ 1 })
  local lengths = torch.Tensor({ 5 })

  local noiseCount = 2

  local opts = { lengths=lengths, noiseCount=noiseCount }
  local newData, _ = makePermutationNegativeExamples(data, labels, opts)

  assert(verifyNumberOfElementsModified(data, newData, noiseCount))
  assert(verifyModifiedElementsAreContiguous(data, newData, noiseCount))
  assert(verifyPaddingIsUnchanged(data, newData, 2))
end

testNoiseFractionPermutation = function()
  local data = torch.Tensor({ { 0, 0, 1, 2, 3, 4, 5, 0, 0 } })
  local labels = torch.Tensor({ 1 })
  local lengths = torch.Tensor({ 5 })

  local noiseFraction = .75
  local noiseCount = math.floor(5 * noiseFraction) 

  local opts = { lengths=lengths, noiseFraction=noiseFraction }
  local newData, _ = makePermutationNegativeExamples(data, labels, opts)

  assert(verifyNumberOfElementsModified(data, newData, noiseCount))
  assert(verifyModifiedElementsAreContiguous(data, newData, noiseCount))
  assert(verifyPaddingIsUnchanged(data, newData, 2))
end

testNoiseCountReplacement = function()
  local data = torch.Tensor({ { 0, 0, 100, 200, 300, 400, 500, 0, 0 } })
  local labels = torch.Tensor({ 1 })
  local lengths = torch.Tensor({ 5 })

  local noiseCount = 2

  local opts = { lengths=lengths, noiseCount=noiseCount }
  local newData, _ = makeReplacementNegativeExamples(data, labels, opts)

  assert(verifyNumberOfElementsModified(data, newData, noiseCount))
  assert(verifyPaddingIsUnchanged(data, newData, 2))
end

testNoiseFractionReplacement = function()
  local data = torch.Tensor({ { 0, 0, 100, 200, 300, 400, 500, 0, 0 } })
  local labels = torch.Tensor({ 1 })
  local lengths = torch.Tensor({ 5 })

  local noiseFraction = .75
  local noiseCount = math.floor(5 * noiseFraction) 

  local opts = { lengths=lengths, noiseFraction=noiseFraction }
  local newData, _ = makePermutationNegativeExamples(data, labels, opts)

  assert(verifyNumberOfElementsModified(data, newData, noiseCount))
  assert(verifyPaddingIsUnchanged(data, newData, 2))
end


runTests = function()
  for i=1,200 do
    testNoiseCountPermutation()
    testNoiseFractionPermutation()
    testNoiseCountReplacement()
    testNoiseFractionReplacement()
  end
end

local rngState = torch.getRNGState()
torch.manualSeed(1)

local succeeded, err = pcall(runTests)
if not succeeded then
  print(err)
end

torch.setRNGState(rngState)
