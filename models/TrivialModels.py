import numpy as np
import torch
from FNN_surrogate_nested import Surrogate

torch.set_default_tensor_type(torch.DoubleTensor)
class circuitTrivial:
    def __init__(self):
        # Init parameters
        self.defParam = torch.Tensor([[3.0, 5.0]])
        self.RM = torch.Tensor([[1.0, 1.0],
                                [1.0, -1.0]])
        self.stdRatio = 0.05
        self.surrogate = Surrogate("Trivial", self.solve_t, 2, 2, [[0, 6], [0, 6]], 20)
        self.data = None

    def genDataFile(self, dataSize=50, dataFileName="source/data/data_trivial.txt", store=True):
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

    def evalNegLL_t(self, params, surrogate=True):
        stds = torch.abs(self.solve_t(self.defParam)) * self.stdRatio
        if not surrogate:
            modelOut = self.solve_t(params)
        else:
            modelOut = self.surrogate.forward(params)
        Data = torch.tensor(self.data)
        # Eval LL
        ll1 = -0.5 * np.prod(self.data.shape) * np.log(2.0 * np.pi)
        ll2 = (-0.5 * self.data.shape[1] * torch.log(torch.prod(stds))).item()
        ll3 = - 0.5 * torch.sum(torch.sum((modelOut.unsqueeze(0) - Data.t().unsqueeze(1)) ** 2, dim=0) / stds[0] ** 2, dim=1, keepdim=True)
        negLL = -(ll1 + ll2 + ll3)
        return negLL

    def den_t(self, params, surrogate=True):
        return - self.evalNegLL_t(params, surrogate)