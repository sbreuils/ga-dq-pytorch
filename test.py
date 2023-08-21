import torch
import torchdualquat as dqs 
import math
import numpy as np

dq1 = dqs.Quaternion(torch.tensor([[1,0,1,0],[1,0,0,0],[0,1.0,0.,0.]]))
dq2 = 3.*dqs.Quaternion(torch.tensor([1,0,0,0]))
dq3 = dq1.conjugate()

# dq4 = 2*dqs.Quaternion(torch.tensor([math.cos(math.pi/4),math.sin(math.pi/4),0,0]))
# dq4 = dqs.Quaternion(3*torch.tensor([[math.cos(math.pi/4),math.sin(math.pi/4),0,0],[math.cos(math.pi/4),0,math.sin(math.pi/4),0]]))

# dq4 = dqs.DualQuaternion(dqs.Quaternion(3*torch.tensor([[math.cos(math.pi/4),math.sin(math.pi/4),0,0],[math.cos(math.pi/4),0,math.sin(math.pi/4),0]])),)
dq4 = dqs.DualQuaternion.zeroDualQuaternion()
dq5 = dqs.DualQuaternion.idDualQuaternion()
dq6 = 2*(dq5+dq4)

dq7 = dq5
dq7 *= dq5

print(dq4)
print(dq5)
print(dq6)
print(dq7.dagger())


N=8
vecOfRotationsAxisAnglesRot = torch.from_numpy(np.expand_dims(np.linspace(0,2*torch.pi,N),axis=1)*(np.array( N*[1.,0,0]).reshape(N,-1))).double() 
qPure = torch.from_numpy(np.expand_dims(np.linspace(0,np.sqrt(2)/2,N),axis=1)*(np.array(N*[1,1,1]).reshape(N,-1))).double()


dqTarget = dqs.DualQuaternion.fromAxisAngle_Translation(vecOfRotationsAxisAnglesRot,qPure)

print(qPure)

# print("trans quat new shape=",qtrans)


# print(dq4.normalize())



# print(dq1)
# print(dq2)
# print(dq3)