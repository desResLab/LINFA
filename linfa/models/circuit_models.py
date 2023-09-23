import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

def solve_ivp_s(func, t0, t_bound, y0, max_step, t_eval, batch_size, aux_size, device='cpu'):
    # PyTorch - Reimplementing scipy.integrate.solve_ivp (Runge-Kuta Method solving ODE) - Output compared and tested.
    n = int((t_bound - t0) / max_step)
    t = t0
    y = y0.double()
    y_rec = torch.zeros(len(t_eval), batch_size).to(device)
    aux_rec = torch.zeros(len(t_eval), aux_size, batch_size).to(device)
    t_rec = torch.zeros(len(t_eval)).to(device)
    i = 0
    for _ in range(n):
        res, aux = func(t, y)
        k1 = max_step * res
        res, aux = func(t + 0.5 * max_step, y + 0.5 * k1)
        k2 = max_step * res
        res, aux = func(t + 0.5 * max_step, y + 0.5 * k2)
        k3 = max_step * res
        res, aux = func(t + 0.5 * max_step, y + k3)  # I think this should be 1.0 and not 0.5!!!
        k4 = max_step * res
        delta = 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        if t >= t_eval[i]:
            y_rec[i, :] = y
            aux_rec[i, :, :] = aux
            t_rec[i] = t
            i = i + 1
        y = y + delta
        t = t + max_step

    y_rec[len(t_eval) - 1, :] = y
    aux_rec[len(t_eval) - 1, :, :] = aux
    t_rec[len(t_eval) - 1] = t
    return t_rec, y_rec, aux_rec


def trapz(t, y):
    # Pytorch - Numerical Integration: Trapezoidal Rule
    S = list(y.size())
    t0 = t[:S[0] - 1]
    t1 = t[1:]
    y0 = y[:S[0] - 1, :]
    y1 = y[1:, :]
    return torch.sum((y1 + y0) / 2.0 * (t1 - t0).reshape(S[0] - 1, 1), 0)


class circuitModel():

    def __init__(self, numParam, numState, numAuxState, numOutputs,
                 parName, limits, defParam,
                 cycleTime, totalCycles, forcing=None, device='cpu'):
        
        # Assign device
        self.device = device
        # Time integration parameters
        self.cycleTime = cycleTime
        self.totalCycles = totalCycles
        # Forcing
        self.forcing = forcing
        # Init parameters
        self.numParam = numParam
        self.numState = numState
        self.numAuxState = numAuxState
        self.numOutputs = numOutputs
        self.parName = parName

        self.stdRatio = 0.05
        self.limits = limits
        self.mmHgToBarye = 1333.22
        self.defParam = defParam
        self.defOut = self.solve_t(self.defParam, y0=None)
        self.data = None

    def evalDeriv_t(self, t, y, params):
        # PyTorch - Computing Derivative for the model. See specific definition in subclasses of RC and RCR
        pass

    def postProcess_t(self, t, y, aux, start, stop):
        # PyTorch - Computing (min, max, ave) output given solution of ODE. See specific definition in subclasses of RC and RCR
        pass

    def genDataFile(self, dataSize, dataFileName):
        # Scipy - Generate Data file: Given the solution def_out of default parameters,
        # sample data with mean def_out and cov matrix with diagonal std * def_out
        data = np.zeros((self.numOutputs, dataSize))
        # Get Standard Deviaitons using ratios
        stds = self.defOut * self.stdRatio
        for loopA in range(dataSize):
            # Get Default Paramters
            data[:, loopA] = self.defOut[0] + torch.randn(len(stds[0])) * stds
        self.data = data
        np.savetxt(dataFileName, data)

    def solve_t(self, params, y0=None):
        # Pytorch - Reimplementing solve: Support Multiple parameters
        batch_size = list(params.size())[0]
        if y0 is None: y0 = 55.0 * self.mmHgToBarye * torch.ones(batch_size).to(self.device)
        t_bound = self.totalCycles * self.cycleTime
        saveSteps = np.linspace(0.0, t_bound, 201, endpoint=True)
        odeSol_t, odeSol_y, odeSol_aux = solve_ivp_s(lambda t, y: self.evalDeriv_t(t, y, params.double()), 0.0, t_bound,
                                                     y0 * torch.ones(batch_size).to(self.device),
                                                     max_step=self.cycleTime / 1000.0, t_eval=saveSteps,
                                                     batch_size=batch_size, aux_size=self.numAuxState, device=self.device)
        start = len(saveSteps) - (len(saveSteps[saveSteps > (self.totalCycles - 1) * self.cycleTime]) + 1)
        stop  = len(saveSteps)

        return self.postProcess_t(odeSol_t, odeSol_y, odeSol_aux, start, stop)

class rcModel(circuitModel):

    def __init__(self, cycleTime, totalCycles, forcing=None, device='cpu'):
        # Init parameters
        numParam = 2
        numState = 1
        numAuxState = 4
        numOutputs = 3
        parName = ["R", "C"]
        limits = torch.Tensor([[100.0, 1500.0], [1.0e-5, 1.0e-2]]).to(device)
        defParam = torch.Tensor([[1000.0, 0.00005]]).to(device)
        #  Invoke Superclass Constructor
        super().__init__(numParam, numState, numAuxState, numOutputs,
                         parName, limits, defParam,
                         cycleTime, totalCycles, forcing, device=device)

    def evalDeriv_t(self, t, y, params):
        # Pytorch - Evaluate Derivative.
        R = params[:, 0]
        C = params[:, 1]
        Pd = torch.tensor(55 * self.mmHgToBarye).to(self.device)
        P1 = y

        # Interpolate forcing
        Q1 = np.interp(t % self.cycleTime, self.forcing[:, 0], self.forcing[:, 1])
        Q2 = (P1 - Pd) / R
        dP1dt = (Q1 - Q2) / C

        aux = torch.zeros(tuple((self.numAuxState,)) + tuple(dP1dt.size()))
        aux[0] = Pd
        aux[1] = Q1
        aux[2] = Q2
        return dP1dt, aux

    def postProcess_t(self, t, y, aux, start, stop):
        # PyTorch - Computing (Min, Max, Ave) tuple given solution of ODE
        return torch.stack([torch.min(y[start:stop, :], 0)[0] / self.mmHgToBarye,
                            torch.max(y[start:stop, :], 0)[0] / self.mmHgToBarye,
                            trapz(t[start:stop], y[start:stop, :]) / float(self.cycleTime) / self.mmHgToBarye],
                           dim=1)

class rcrModel(circuitModel):

    def __init__(self, cycleTime, totalCycles, forcing=None, device='cpu'):
        # Init parameters
        numParam = 3
        numState = 1
        numAuxState = 4
        numOutputs = 3
        parName = ["R1", "R2", "C"]
        limits = torch.Tensor([[100.0, 1500.0],
                               [100.0, 1500.0],
                               [1.0e-5, 1.0e-2]]).to(device)
        defParam = torch.Tensor([[1000.0, 1000.0, 0.00005]]).to(device)
        #  Invoke Superclass Constructor
        super().__init__(numParam, numState, numAuxState, numOutputs,
                         parName, limits, defParam,
                         cycleTime, totalCycles, forcing, device=device)
        self.stdRatio = 0.05

    def evalDeriv_t(self, t, y, params):
        R1 = params[:, 0]
        R2 = params[:, 1]
        C = params[:, 2]
        Pd = 55 * self.mmHgToBarye * torch.ones(params.shape[0]).to(self.device)
        P1 = y

        # Interpolate forcing
        Q1 = np.interp(t % self.cycleTime, self.forcing[:, 0], self.forcing[:, 1])
        P0 = P1 + R1 * Q1
        Q2 = (P1 - Pd) / R2
        dP1dt = (Q1 - Q2) / C

        aux = torch.zeros(tuple((self.numAuxState,)) + tuple(dP1dt.size()))
        aux[0] = Pd
        aux[1] = P0
        aux[2] = Q1
        aux[3] = Q2
        return dP1dt, aux

    def postProcess_t(self, t, y, aux, start, stop):
        return torch.stack([torch.min(aux[start:stop, 1, :], 0)[0] / self.mmHgToBarye,
                            torch.max(aux[start:stop, 1, :], 0)[0] / self.mmHgToBarye,
                            trapz(t[start:stop], aux[start:stop, 1, :]) / float(self.cycleTime) / self.mmHgToBarye],
                           dim=1)

# GEN DATA
if __name__ == '__main__':
  
  # model = rcModel()
  model = rcrModel()
  model.genDataFile(50, "../../resource/data/data_rcr.txt")