
-- test max pooling

-- libs:
require 'sys'   -- timer
require 'cunn'  -- cuda

-- dev:
cutorch.setDevice(arg[1] or 1)

print('DEVID = ' .. cutorch.getDevice())

-- layout
mb = 128
ic = 32
ix = 64
iy = 64

-- batch major (torch)
input1 = torch.randn(mb,ic,ix,iy)
input1 = input1:cuda()
net1 = nn.Sequential()
net1:add(nn.SpatialMaxPooling(2,2,2,2))
net1:cuda()
net1:forward(input1)
net1:backward(input1, net1.output)
cutorch.synchronize()

-- feature major (alex)
input2 = torch.randn(ic,ix,iy,mb)
input2 = input2:cuda()
-- input2 = input1:transpose(1,4)
-- input2 = input1
net2 = nn.Sequential()
--net2:add( nn.Transpose( {1,2},{2,3},{3,4}) )
net2:add(nn.SpatialMaxPoolingCUDA(2,2,2,2))
--net2:add( nn.Transpose( {3,4},{2,3},{1,2}) )
net2:cuda()
net2:forward(input2)
net2:backward(input2, net2.output)
cutorch.synchronize()

-- benchmark
function benchmark(n,i)
   nbOfAverages = 3
   sys.tic()
   for t = 1,nbOfAverages do
      n:updateOutput(i)
      n:updateGradInput(i, n.output)
      n:accGradParameters(i, n.output)
   end
   cutorch.synchronize()
   t = sys.toc()/nbOfAverages
   print('CUDA - Time(s):' ..t)
end

benchmark(net1,input1)
benchmark(net2,input2)

-- compare results
--print((net1.output:float()-net2.output:float()):norm())
--print((net1.gradInput:float()-net2.gradInput:float()):norm())
--print(input2)
--print(net2.output)
--print(net2.gradInput)
