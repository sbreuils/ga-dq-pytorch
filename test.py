import torch
import dqs 


dq1 = dqs.Quaternion(torch.tensor([[1,0,1,0],[1,0,0,0],[0,1.0,0.,0.]]))
dq2 = 3.*dqs.Quaternion(torch.tensor([1,0,0,0]))
dq3 = dq1.conjugate()


print(dq1)
print(dq2)
print(dq3)