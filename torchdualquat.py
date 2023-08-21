import torch
import numpy as np
import math

from torchquat import *


"""
Based on https://gist.github.com/Flunzmas/d9485d9fee6244b544e7e75bdc0c352c
and https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py
"""

class DualQuaternion:
    def __init__(self,qr,qd):
        '''
            Dual Quaternion Q from 2 quaternions
            DQ = qr + epsilon qd
        '''
        # assert qr.shape==qd.shape
        self.qr = qr
        self.qd = qd
    
    @classmethod
    def idDualQuaternion(cls):
        return cls(Quaternion.idQuaternion(),Quaternion.zeroQuaternion())
    
    @classmethod
    def zeroDualQuaternion(cls):
        return cls(Quaternion.zeroQuaternion(),Quaternion.zeroQuaternion())

    @classmethod
    def zerosDualQuaternions(cls,N):
        return cls(Quaternion.zerosQuaternion(N),Quaternion.zerosQuaternion(N))
    
    @classmethod
    def fromAxisAngle_Translation(cls,axis_angle,translation):
        '''
        Convert rotations given as axis/angle and translations to dual quaternions.
        axis/angle shape is a [... , 3] tensor
        translation shape is a [... , 3] tensor
        '''
        assert axis_angle.shape[-1]==3
        assert translation.shape[-1]==3

        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = angles * 0.5
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        qr = Quaternion(torch.cat(
            [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
        ))
        
        trans_size = list(translation.shape)
        trans_size[-1] = 4
        qtrans = torch.zeros(trans_size)
        qtrans[:,1:]=translation

        qd = 0.5 * Quaternion(qtrans) * qr

        return DualQuaternion(qr,qd)



    def __mul__(self,other):
        '''
            self*other 
            we assume self and other are dual quaternions

        '''
        dq_prod_r = self.qr*other.qr
        dq_prod_d = self.qr*other.qd + self.qd*other.qr
        return DualQuaternion(dq_prod_r,dq_prod_d)

    def __imul__(self,other):
        '''
            self*=other 
            we assume self is dual quaternions and other is a scalar
        '''
        return self.__mul__(other)

    def __rmul__(self, other):
        '''Multiplication with a scalar
        '''
        return DualQuaternion(other*self.qr,other*self.qd)

    def __add__(self,other):
        '''
            self+other 
            we assume self and other are dual quaternions
        '''
        return DualQuaternion(self.qr+other.qr,self.qd+other.qd)

    def translation_part(self):
        '''
        Returns the translation component of the input dual quaternion.
        Translation is returned as tensor of shape [*, 3].
        '''

        mult = (2.0*self.qd)*self.qr.conjugate()

        return mult.q[..., 1:]
    
    def conjugate(self):
        '''
        DQ * = Qr* + epsilon Qd*
        '''

        return DualQuaternion(self.qr.conjugate(),self.qd.conjugate())


    def dagger(self):
        """
        DQ dagger = Qr* - epsilon Qd*
        This form is commonly used to transform a point
        """
        return DualQuaternion(self.qr.conjugate(),-1*self.qd.conjugate())

    def normalize(self):
        """
        Normalize the coefficients of a given dual quaternion tensor .
        """
        norm = torch.sqrt(torch.sum(torch.square(self.qr.q), dim=-1))  
        assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=self.qr.device)))  # check for singularities
        return DualQuaternion(torch.div(self.qr.q, norm.unsqueeze(-1)),torch.div(self.qd.q, norm.unsqueeze(-1)))  # dq_norm = dq / ||q|| = dq_r / ||dq_r|| + dq_d / ||dq_r||







    def __str__(self):
        return str(self.qr)+' + epsilon *'+ str(self.qd)

if __name__ == "__main__":
    # test quaternion and dual quaternion methods
    import doctest
    doctest.testmod(verbose = True)



