import torch
from model import NetD, NetG
from torch.autograd import Variable

# Check NetG
tfidf = Variable(torch.rand(4, 11083))
z = Variable(torch.rand(4, 200))
netG = NetG()
out = netG(z, tfidf)
print(out)
print('NetG works properly')

# Check NetD
feat = Variable(torch.rand(4, 3584))
netD = NetD()
o1, o2 = netD(feat)
print(o1)
print(o2)
print('NetD works properly')
