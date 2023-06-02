import os
import torch
import numpy as np
import scipy as sp

class Transformation(torch.nn.Module):

    def __init__(self, func_info=None):
        super().__init__()
        self.funcs     = []
        self.log_jacob = []
        self.n = len(func_info)
        for func, a, b, c, d in func_info:
            if func == "identity":
                self.funcs.append(lambda x: x)
                self.log_jacob.append(lambda x: 0.0)
            elif func == "tanh":
                m1 = (a + b) / 2
                t1 = (b - a) / 2
                m2 = (c + d) / 2
                t2 = (d - c) / 2
                self.funcs.append(lambda x: torch.tanh((x - m1) / (b - a) * 6.0) * t2 + m2)
                self.log_jacob.append(lambda x: torch.log(1.0 - torch.tanh((x - m1) / (b - a) * 6.0) ** 2) + np.log(t2) + np.log(3.0) - np.log(t1))
            elif func == "linear":
                self.funcs.append(lambda x: (x - a) / (b - a) * (d - c) + c)
                self.log_jacob.append(lambda x: np.log(d - c) - np.log(b - a))
            elif func == "exp":
                self.funcs.append(lambda x: torch.exp((x - a) / (b - a) * (np.log(d) - np.log(c)) + np.log(c)))
                self.log_jacob.append(lambda x: (x - a) / (b - a) * (np.log(d) - np.log(c)) + np.log(c) + np.log(np.log(d) - np.log(c)) - np.log(b - a))

    def forward(self, z):
        """
        
        from desired scale to original scale, evaluate original samples.

        Args:
            z (torch.Tensor): samples in desired scale.

        Returns:
            torch.Tensor: samples in original scale.
        """
        if z.size(1) != self.n:
            raise ValueError("Inconsistent size. Got {}, should be {}".format(z.size(1), self.n))
        zi = torch.chunk(z, chunks=self.n, dim=1)
        x = []
        for zz, func in zip(zi, self.funcs):
            x.append(func(zz))

        return torch.cat(x, dim=1)


    def compute_log_jacob_func(self, z):
        """
        
        from desired scale to original scale, compute log absolute determinant of Jacobian matrix.

        Args:
            z (torch.Tensor): samples in desired scale.

        Returns:
            torch.Tensor: samples in original scale.
        """
        if z.size(1) != self.n:
            raise ValueError("Inconsistent size. Got {}, should be {}".format(z.size(1), self.n))
        zi = torch.chunk(z, chunks=self.n, dim=1)
        jacob_res = 0
        for zz, log_jacob_func in zip(zi, self.log_jacob):
            jacob_res += log_jacob_func(zz)

        return jacob_res
