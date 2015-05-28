require 'nn'

nWords = 1000
wordDims = 100
nFilters = 10
filterWidth = 3
stride = 1

lookup = nn.LookupTable(nWords, wordDims)
conv = nn.TemporalConvolution(wordDims, nFilters, filterWidth, stride)
bmm = nn.MM()

input = (torch.rand(10) * nWords):long() + 1
sentenceMatrix = lookup:forward(input)
bmm:forward(conv.weight, 
