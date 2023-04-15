import sys, os
import numpy as np
import torch
from FNN_surrogate_nested import Surrogate
torch.set_default_tensor_type(torch.DoubleTensor)


class Highdim:
    def __init__(self):
        # Init parameters
        self.input_num = 5
        self.output_num = 4
        self.x0 = torch.Tensor([0.0838, 0.2290, 0.9133, 0.1524, 0.8258])
        self.defParam = torch.Tensor([[15.6426, 0.2231, 1.2840, 0.0821, 5.7546]])
        self.RM = torch.Tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]) / np.sqrt(2.0)
        self.defOut = self.solve_t(self.defParam)
        self.stdRatio = 0.01
        self.data = None
        self.surrogate = Surrogate("highdim", lambda x: self.solve_t(self.transform(x)), self.input_num, self.output_num,
                                   torch.Tensor([[-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]]), 20)


    def genDataFile(self, dataSize=50, dataFileName="source/data/data_highdim.txt", store=True):
        def_out = self.defOut[0]
        self.data = def_out + self.stdRatio * torch.abs(def_out) * torch.normal(0, 1, size=(dataSize, self.output_num))
        self.data = self.data.t().detach().numpy()
        if store: np.savetxt(dataFileName, self.data)
        return self.data

    def solve_t(self, params):
        return torch.matmul((2 * torch.abs(2 * self.x0 - 1) + params) / (1 + params), self.RM)

    def evalNegLL_t(self, modelOut):
        data_size = len(self.data[0])
        stds = self.defOut * self.stdRatio
        Data = torch.tensor(self.data)
        ll1 = -0.5 * np.prod(self.data.shape) * np.log(2.0 * np.pi)  # a number
        ll2 = (-0.5 * self.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
        ll3 = - 0.5 * torch.sum(torch.sum((modelOut.unsqueeze(0) - Data.t().unsqueeze(1)) ** 2, dim=0) / stds[0] ** 2, dim=1, keepdim=True)
        negLL = -(ll1 + ll2 + ll3)
        return negLL

    def transform(self, x):
        return torch.exp(x)

    def den_t(self, x, surrogate=True):
        batch_size = x.size(0)
        adjust = torch.sum(x, dim=1, keepdim=True)
        if surrogate:
            modelOut = self.surrogate.forward(x)
        else:
            modelOut = self.solve_t(self.transform(x))
        return - self.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust
    def rev_solve_t(self, y):
        x = torch.Tensor([[0]] * y.size(0))
        x = torch.cat([y[:, 3:4], x], dim=1)
        x = torch.cat([y[:, 2:3] - x[:, 0:1], x], dim=1)
        x = torch.cat([y[:, 1:2] - x[:, 0:1], x], dim=1)
        x = torch.cat([y[:, 0:1] - x[:, 0:1], x], dim=1) * np.sqrt(2)

        con = 2 * torch.abs(2 * self.x0 - 1)
        print(con)
        t1 = (1 - x) / torch.Tensor([1, -1, 1, -1, 1])
        t2 = (con - x) / torch.Tensor([1, -1, 1, -1, 1])
        tmin = torch.cat([t1[:, 0:1], t2[:, 1:2], t1[:, 2:3], t2[:, 3:4], t1[:, 4:5]], dim=1)
        tmax = torch.cat([t2[:, 0:1], t1[:, 1:2], t2[:, 2:3], t1[:, 3:4], t2[:, 4:5]], dim=1)
        tmin = torch.max(tmin, dim=1, keepdim=True)[0]
        tmax = torch.min(tmax, dim=1, keepdim=True)[0]
        return x, torch.cat([tmin, tmax], dim=1)