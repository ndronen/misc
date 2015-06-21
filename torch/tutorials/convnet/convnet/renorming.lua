--[[
Example of limiting the 2-norms of the word-representation portions of
convolutional kernels to some constant.  `conv` is a TemporalConvolution
layer with 1 kernel, kw=2, inputFrameSize=5, so each kernel is a 1x10
matrix.
--]]
th> w1 = torch.Tensor()
th> w1:set(conv.weight:storage(), 1, torch.LongStorage({1, 5}))
th> w1:renorm(2, 1, 1)
th> w1:norm(2, 2)
th> w1:set(conv.weight:storage(), 6, torch.LongStorage({1, 5}))
th> w1:renorm(2, 1, 1)
th> w1:norm(2, 2)
th> conv.weight:norm(2, 2)
