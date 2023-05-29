import numpy as np
import torch

# from FNN_surrogate_nested import Surrogate

torch.set_default_tensor_type(torch.DoubleTensor)


def solve_ivp_s(func, t0, t_bound, y0, max_step, t_eval, batch_size, aux_size):
    # PyTorch - Reimplementing scipy.integrate.solve_ivp (Runge-Kuta Method solving ODE) - Output compared and tested.
    n = int((t_bound - t0) / max_step)
    t = t0
    y = y0.double()
    y_rec = torch.zeros(len(t_eval), batch_size)
    aux_rec = torch.zeros(len(t_eval), aux_size, batch_size)
    t_rec = torch.zeros(len(t_eval))
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
                 cycleTime, totalCycles, forcing=None):
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

        self.stdRatio = 0.01
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
        if y0 is None: y0 = 55.0 * self.mmHgToBarye * torch.ones(batch_size)
        t_bound = self.totalCycles * self.cycleTime
        saveSteps = np.linspace(0.0, t_bound, 201, endpoint=True)
        odeSol_t, odeSol_y, odeSol_aux = solve_ivp_s(lambda t, y: self.evalDeriv_t(t, y, params.double()), 0.0, t_bound,
                                                     y0 * torch.ones(batch_size).double(),
                                                     max_step=self.cycleTime / 1000.0, t_eval=saveSteps,
                                                     batch_size=batch_size, aux_size=self.numAuxState)
        start = len(saveSteps) - (len(saveSteps[saveSteps > (self.totalCycles - 1) * self.cycleTime]) + 1)
        stop = len(saveSteps)

        return self.postProcess_t(odeSol_t, odeSol_y, odeSol_aux, start, stop)

    def evalNegLL_t(self, modelOut):
        # PyTorch - Evaluate Negative Log-Likelihood with multiple parameters.
        data_size = len(self.data[0])
        # Get the absolute values of the standard deviations
        stds = self.defOut * self.stdRatio
        Data = torch.tensor(self.data)
        # Eval LL
        ll1 = -0.5 * np.prod(self.data.shape) * np.log(2.0 * np.pi)  # a number
        ll2 = (-0.5 * self.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
        ll3 = 0.0
        for i in range(3):
            ll3 = ll3 - 0.5 * torch.sum(
                ((modelOut[:, i].repeat(data_size, 1).t().float() - Data[i, :].float()) / stds[0, i]) ** 2, dim=1)
        negLL = -(ll1 + ll2 + ll3)
        return negLL

    def den_t(self, xx, surrogate=None):
        # PyTorch - True Log Posterior of Model.
        pass


class rcModel(circuitModel):
    def __init__(self, cycleTime, totalCycles, forcing=None):
        # Init parameters
        numParam = 2
        numState = 1
        numAuxState = 4
        numOutputs = 3
        parName = ["R", "C"]
        limits = torch.Tensor([[100.0, 1500.0], [1.0e-5, 1.0e-2]])
        defParam = torch.Tensor([[1000.0, 0.00005]])
        #  Invoke Superclass Constructor
        super().__init__(numParam, numState, numAuxState, numOutputs,
                         parName, limits, defParam,
                         cycleTime, totalCycles, forcing)
        # self.surrogate = Surrogate("RC", lambda x: self.solve_t(self.transform(x)), numParam, numOutputs,
        #                            torch.Tensor([[-7, 7], [-7, 7]]), 20)

    def evalDeriv_t(self, t, y, params):
        # Pytorch - Evaluate Derivative.
        R = params[:, 0]
        C = params[:, 1]
        Pd = 55 * self.mmHgToBarye
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

    def transform(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        return torch.cat((torch.tanh(x1 / 7.0 * 3.0) * 700.0 + 800.0, torch.exp(x2 / 7.0 * 3.0 - 8.0)), 1)

    def den_t(self, x, surrogate=True):
        batch_size = list(x.size())[0]
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        adjust = torch.log(1.0 - torch.tanh(x1 / 7.0 * 3.0) ** 2) + x2 / 7 * 3
        if surrogate:
            modelOut = self.surrogate.forward(x)
        else:
            modelOut = self.solve_t(self.transform(x))
        return - self.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust


class rcrModel(circuitModel):
    def __init__(self, cycleTime, totalCycles, forcing=None):
        # Init parameters
        numParam = 3
        numState = 1
        numAuxState = 4
        numOutputs = 3
        parName = ["R1", "R2", "C"]
        limits = torch.Tensor([[100.0, 1500.0],
                               [100.0, 1500.0],
                               [1.0e-5, 1.0e-2]])
        defParam = torch.Tensor([[1000.0, 1000.0, 0.00005]])
        #  Invoke Superclass Constructor
        super().__init__(numParam, numState, numAuxState, numOutputs,
                         parName, limits, defParam,
                         cycleTime, totalCycles, forcing)
        # self.surrogate = Surrogate("RCR", lambda x: self.solve_t(self.transform(x)), numParam, numOutputs,
        #                            torch.Tensor([[-7, 7], [-7, 7], [-7, 7]]), 20)

    def evalDeriv_t(self, t, y, params):
        R1 = params[:, 0]
        R2 = params[:, 1]
        C = params[:, 2]
        Pd = 55 * self.mmHgToBarye * torch.ones(params.shape[0])
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

    def transform(self, x):
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
        return torch.cat((torch.tanh(x1 / 7.0 * 3.0) * 700.0 + 800.0,
                          torch.tanh(x2 / 7.0 * 3.0) * 700.0 + 800.0,
                          torch.exp(x3 / 7.0 * 3.0 - 8.0)), 1)

    def den_t(self, x, surrogate=True):
        batch_size = list(x.size())[0]
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
        adjust = torch.log(1.0 - torch.tanh(x1 / 7.0 * 3.0) ** 2) \
                 + torch.log(1.0 - torch.tanh(x2 / 7.0 * 3.0) ** 2) \
                 + x3 / 7 * 3
        if surrogate:
            modelOut = self.surrogate.forward(x)
        else:
            modelOut = self.solve_t(self.transform(x))
        return - self.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust
