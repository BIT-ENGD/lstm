import torch
t=[0]*10
print(type(t))

print(torch.eye(2,3))
print(torch.cuda.get_device_capability(0))

x=torch.randn(3,3)
print(x)
m=x.max(1)
print(m[1])


T=torch.randn(2,3)
print(T)

S=T.unsqueeze(2)
print(S)
print(S.squeeze(0))


import torch.nn as nn
TT=nn.Embedding(6,3)

x=[[2,3,5],[1,2,3]]
s=TT(torch.LongTensor(x))
print(s)

