import sys, os
import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

class Highdim:

    def __init__(self,device='cpu'):
        # Init parameters
        self.input_num = 5
        self.output_num = 4
        self.x0 = torch.Tensor([0.0838, 0.2290, 0.9133, 0.1524, 0.8258]).to(device)
        self.defParam = torch.Tensor([[2.75, -1.5, 0.25, -2.5, 1.75]]).to(device)
        self.RM = torch.Tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]).to(device) / np.sqrt(2.0)
        self.defOut = self.solve_t(self.defParam)
        self.stdRatio = 0.01
        self.data = None

    def genDataFile(self, dataSize=50, dataFileName="../../resource/data/data_highdim.txt", store=True):
        def_out = self.defOut[0]
        self.data = def_out + self.stdRatio * torch.abs(def_out) * torch.normal(0, 1, size=(dataSize, self.output_num))
        self.data = self.data.t().detach().numpy()
        if store: np.savetxt(dataFileName, self.data)
        return self.data

    def solve_t(self, params):
        return torch.matmul((2 * torch.abs(2 * self.x0 - 1) + torch.exp(params)) / (1 + torch.exp(params)), self.RM)

# GEN DATA
if __name__ == '__main__':
  
  model = Highdim()
  model.genDataFile(50)