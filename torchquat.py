import torch
import numpy as np
import math




"""
Based on https://gist.github.com/Flunzmas/d9485d9fee6244b544e7e75bdc0c352c
and https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py
"""

class Quaternion:
    def __init__(self,q):
        '''
            Quaternion Q from a torch array [[w,v1,v2,v3]]
            Q = w + v = w + v1*i + v2*j + v3*k
        '''
        assert q.shape[-1] == 4
        self.q = q

    @classmethod
    def quaternionFromComponents(cls,w,v1,v2,v3):
        '''
            Q = w + v = w + v1*i + v2*j + v3*k
        '''
        return cls(torch.tensor([[w,v1,v2,v3]]))
    
    @classmethod
    def idQuaternion(cls):
        '''
            Q = w + v = 1 + 0*i + 0*j + 0*k
        '''
        return cls(torch.tensor([[1.,0.,0.,0.]]))

    @classmethod
    def zeroQuaternion(cls):
        '''
            Q = w + v = 0 + 0*i + 0*j + 0*k
        '''
        return cls(torch.tensor([[0.,0.,0.,0.]]))

    @classmethod
    def zerosQuaternion(cls,N):
        '''
            Q = [0 + 0*i + 0*j + 0*k]*N
        '''
        return cls(torch.zeros(N, 4))

    def __add__(self,other):
        '''
            self+other 
            we assume self and other are quaternions
        '''
        assert other.q.shape[-1] == 4
        return Quaternion(self.q+other.q)

    def __mul__(self,other):
        '''
            self*other 
            we assume self and other has 4 columns 
            >>> (Quaternion(torch.tensor([math.cos(torch.pi/4),math.sin(math.pi/4),0,0]))*Quaternion(torch.tensor([math.cos(math.pi/4),math.sin(math.pi/4),0,0]))).q
            tensor([0.0000, 1.0000, 0.0000, 0.0000])
        '''
        assert other.q.shape[-1] == 4
        original_shape = self.q.shape

        # Compute outer product
        terms = torch.bmm(other.q.view(-1, 4, 1), self.q.view(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

        return Quaternion(torch.stack((w, x, y, z), dim=1).view(original_shape))
    
    def __imul__(self, other):
        '''
        quaternion multiplication with self-assignment: q1 *= q2
        See __mul__
        '''
        return self.__mul__(other)
    
    def __rmul__(self, other):
        '''Multiplication with a scalar
        :param other: scalar
        >>> (3*Quaternion(torch.tensor([[1,0,0,1]]))).q
        tensor([[3, 0, 0, 3]])
        '''
        return Quaternion(self.q * other)

    def __str__(self):
        return str(self.q)
    
    def shape(self):
        return self.q.shape

    def conjugate(self):
        '''
        conjugate of this quaternion.
        >>> (Quaternion(torch.tensor([[1,0,0,1]]))).conjugate().q
        tensor([[ 1,  0,  0, -1]])
        '''
        assert self.q.shape[-1] == 4
        conj = torch.tensor([1, -1, -1, -1], device=self.q.device)  # multiplication coefficients per element
        return Quaternion(self.q * conj.expand_as(self.q))

    def normalize(self):
        '''
        normalize each quaternion 
        >>> torch.linalg.norm((2*Quaternion(torch.tensor([math.cos(torch.pi/4),math.sin(math.pi/4),0,0]))).normalize().q)
        tensor(1.)
        '''
        assert self.q.shape[-1] == 4
        norm = torch.sqrt(torch.sum(torch.square(self.q), dim=-1)) 
        assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=self.q.device)))  # check for singularities
        return Quaternion(torch.div(self.q, norm.unsqueeze(-1)))  

if __name__ == "__main__":
    # test quaternion and dual quaternion methods
    import doctest
    doctest.testmod(verbose = True)



