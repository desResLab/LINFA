import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

class Trivial:
    def __init__(self,device='cpu'):
        # Init parameters
        self.defParam = torch.Tensor([[3.0, 5.0]]).to(device)
        self.RM = torch.Tensor([[1.0, 1.0],
                                [1.0, -1.0]]).to(device)
        self.stdRatio = 0.05
        self.defOut = self.solve_t(self.defParam)
        self.data = None

    def genDataFile(self, dataSize=50, dataFileName="../../resource/data/data_trivial.txt", store=True):
        def_out = self.solve_t(self.defParam)[0]
        print(def_out)
        self.data = def_out + self.stdRatio * torch.abs(def_out) * torch.normal(0, 1, size=(dataSize, 2))
        self.data = self.data.t().detach().numpy()
        if store: np.savetxt(dataFileName, self.data)
        return self.data

    def solve_t(self, params):
        z1, z2 = torch.chunk(params, chunks=2, dim=1)
        x = torch.cat((z1 ** 3 / 10, torch.exp(z2 / 3)), 1)
        return torch.matmul(x, self.RM)

# GEN DATA
if __name__ == '__main__':
  
  model = Trivial()
  model.genDataFile(50)