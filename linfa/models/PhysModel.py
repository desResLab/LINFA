import numpy as np
import torch

class Phys:
    def __init__(self):
        # define parameter 
        # inputs[0] = 1, [1] = 5, [2] = 60
        self.defParam = torch.Tensor([[1.0, 5.0, 60.0]])

    def genDataFile(self):
        pass

    def solve_t(self):
        pass