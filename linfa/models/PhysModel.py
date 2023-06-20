import numpy as np
import torch

class Phys:
    def __init__(self):
        # define parameter 
        # inputs[0] = 1, [1] = 5, [2] = 60
        self.defParam = torch.Tensor([[1.0, 5.0, 60.0]])
        # self.RM = torch.Tensor([[1.0, 1.0],
        #                         [1.0, -1.0]])
        self.gConst = 9.81
        self.stdRatio = 0.05
        self.data = None

    def genDataFile(self, dataSize = 50, dataFileName="source/data/data_phys.txt", store=True):
        def_out = self.solve_t(self.defParam)[0]
        print(def_out)
        self.data = def_out + self.stdRatio * torch.abs(def_out) * torch.normal(0, 1, size=(dataSize, 3))
        self.data = self.data.t().detach().numpy()
        if store: np.savetxt(dataFileName, self.data)
        return self.data

    def solve_t(self, params):
        z1, z2, z3 = torch.chunk(params, chunks=3, dim=1)
        z3 = z3 * (np.pi / 180)
        # x1: maxHeight
        # x2: finalLocation 
        # x3: totalTime
        x = torch.cat(((z2 ** 2) * (np.sin(z3) ** 2) / (2.0 * self.gConst), 
                    z1 + ((z2 ** 2) * np.sin(2.0 * z3)) / self.gConst, 
                    (2.0 * z2 * np.sin(z3)) / self.gConst), 1)
        return x