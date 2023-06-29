import math
import torch
import numpy as np

class Transformation(torch.nn.Module):

    def __init__(self, func_info=None):
        super().__init__()
        self.funcs     = []
        self.log_jacob = []
        self.n = len(func_info)
        for func, a, b, c, d in func_info:
            if func == "identity":
                self.funcs.append(lambda x: x)
                self.log_jacob.append(lambda x: torch.zeros_like(x))
            elif func == "tanh":
                m1 = (a + b) / 2
                t1 = (b - a) / 2
                m2 = (c + d) / 2
                t2 = (d - c) / 2
                self.funcs.append(lambda x: torch.tanh((x - m1) / (b - a) * 6.0) * t2 + m2)
                self.log_jacob.append(lambda x: torch.log(1.0 - torch.tanh((x - m1) / (b - a) * 6.0) ** 2) + np.log(t2) + np.log(3.0) - np.log(t1))
            elif func == "linear":
                self.funcs.append(lambda x: (x - a) / (b - a) * (d - c) + c)
                self.log_jacob.append(lambda x: torch.tensor([np.log(d - c) - np.log(b - a)]).repeat(x.size(0), 1))
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
    
def test_transform(tets_num):
  
  import matplotlib.pyplot as plt
  
  # Define the transformation params
  if(tets_num == 0):
    a =-7.0
    b = 7.0
    c = 100.0
    d = 1500.0
    trsf_info = [['tanh',a,b,c,d]]
  elif(tets_num == 1):
    a =-7.0
    b = 7.0
    c = math.exp(-8.0)
    d = math.exp(-5.0)
    trsf_info = [['exp',a,b,c,d]]

  # Create transformation
  trsf = Transformation(trsf_info)

  # Create uniform parameterization in [a,b]
  z_vals = torch.from_numpy(np.linspace(a,b,1000)).reshape((-1,1))

  # Eval transformation
  x_vals = trsf.forward(z_vals)
  
  # Eval Jacobian
  x_grad = trsf.compute_log_jacob_func(z_vals)

  # Plot the transformation
  plt.subplot(1,2,1)
  plt.title('Forward map')
  plt.plot(z_vals.numpy(),x_vals.numpy(),'b-')  
  plt.xlim([a,b])
  plt.ylim([c,d])
  plt.subplot(1,2,2)
  plt.title('Jacobian')
  plt.plot(z_vals.numpy(),x_grad.numpy(),'r-')
  plt.xlim([a,b])
  plt.show()

def test_gradient():
  # Set transformation parameters
  # a =-7.0
  a = 0.0
  b = 7.0
  c = 100.0
  d = 1500.0

  trsf_info = [['tanh',a,b,c,d]]
  # trsf_info = [['identity',0,0,0,0]]
  # trsf_info = [['exp',a,b,c,d]]
  # trsf_info = [['linear',a,b,c,d]]

  trsf = Transformation(trsf_info)

  # Create uniform parameterization in [a,b]
  z_vals = torch.from_numpy(np.linspace(a,b,10)).reshape((-1,1))
  z_vals.requires_grad=True
  
  print('z_vals ',z_vals)

  # Eval transformation
  x_vals = trsf.forward(z_vals)

  print('x_vals ',x_vals)
  
  # Eval Jacobian
  x_grad = trsf.compute_log_jacob_func(z_vals)

  x_vals[0].backward()

  print('z_vals.grad ',torch.log(z_vals.grad))
  print('x_grad ',x_grad)

# TEST TRANSFORMATION
if __name__ == '__main__':

  # test_transform(0)
  test_gradient()