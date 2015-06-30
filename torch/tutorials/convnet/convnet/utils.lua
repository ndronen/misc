local json = require 'cjson';
local stringy = require 'stringy';
local fbcunn =  require 'fbcunn';
local kttorch = require 'kttorch';

function fileExists(name)
  local f = io.open(name, "r")
  if f ~= nil then
    io.close(f)
    return true
  else
    return false
  end
end

function loadJson(name)
  local f = io.open(name, 'r')
  local data = f:read()
  return json.decode(data)
end

function incrementIndices(tbl) 
  --[[
  Indices in lua are 1-based.  We have to increment them by one when
  importing from python.
  --]]
  for k,v in pairs(tbl) do
    tbl[k] = v + 1
  end

  return tbl
end

function loadModelInfo(path)
  local modelInfo = loadJson(path)
  modelInfo.index = incrementIndices(modelInfo.index)
  modelInfo.zeroVector = getLargestIndexValue(modelInfo.index)
  return modelInfo
end

function replaceDigits(s)
  return string.gsub(s, "[0-9]", "DIGIT")
end

function separateWordsAndPunctuation(s)
  return string.gsub(s, "(['.,:;?!\"])", " %1 ")
end

function tokenize(s)
  local toks = {}
  local s = replaceDigits(s)
  s = separateWordsAndPunctuation(s)

  for tok in string.gmatch(s, "%S+") do
    table.insert(toks, tok)
  end

  return toks
end

function convertSentenceToIndices(s, modelInfo)
  local indices = {}

  for i, tok in ipairs(tokenize(s)) do
    if modelInfo.index[tok] ~= nil then
      table.insert(indices, modelInfo.index[tok])
    else
      table.insert(indices, modelInfo.zeroVector)
    end
  end

  for i=1,modelInfo.padding do
    table.insert(indices, 1, modelInfo.zeroVector)
    table.insert(indices, modelInfo.zeroVector)
  end

  return torch.Tensor(indices)
end

function getLargestIndexValue(index)
  local maxVal, key = -math.huge

  for k, v in pairs(index) do
    if v > maxVal then
      maxVal, key = v, k
    end
  end

  return maxVal
end

function loadModelAndModelInfo(modelPath, modelInfoPath)
  --[[
  e.g.
  modelPath = 'results/test-wiki/okanohara/1.9m/model.net'
  modelInfoPath = '/tmp/sents-okanohara-wiki-dlm-train-initial-index.json'
  model, modelInfo = loadModel(modelPath, modelInfoPath)
  --]]
  local modelInfo = loadModelInfo(modelInfoPath)
  local model = torch.load(modelPath)
  --[[
  Set the model to evaluate (i.e. test) mode by default so it behaves deterministically.
  --]]
  model:evaluate()
  return model, modelInfo
end
