require 'convnet.data'

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
  end

  return nDiff == n
end

testWindowSizePermutation = function()
  local data = torch.Tensor({ { 0, 0, 1, 2, 3, 4, 5, 0, 0 } })
  local labels = torch.Tensor({ 1 })
  local lengths = torch.Tensor({ 5 })

  local noiseWindowSize = 2

  local opts = { lengths=lengths, noiseWindowSize=noiseWindowSize }
  local newData, _ = makePermutationNegativeExamples(data, labels, opts)

  assert(verifyNumberOfElementsModified(data, newData, noiseWindowSize))
end

testWindowFractionPermutation = function()
  local data = torch.Tensor({ { 0, 0, 1, 2, 3, 4, 5, 0, 0 } })
  local labels = torch.Tensor({ 1 })
  local lengths = torch.Tensor({ 5 })

  local noiseWindowFraction = .75
  local noiseWindowSize = math.floor(5 * noiseWindowFraction) 

  local opts = { lengths=lengths, noiseWindowFraction=noiseWindowFraction }
  local newData, _ = makePermutationNegativeExamples(data, labels, opts)

  assert(verifyNumberOfElementsModified(data, newData, noiseWindowSize))
end

testWindowSizeReplacement = function()
  assert(false)
end

testWindowFractionReplacement = function()
  assert(false)
end

testWindowSizePermutation()
testWindowFractionPermutation()
testWindowSizeReplacement()
testWindowFractionReplacement()
