import torch
import itertools
import numpy as np

class PhysChem(object):

    def __init__(self, var_inputs):
        
        # variable inputs
        self.var_in = torch.tensor(list(itertools.product(*var_inputs)))
        
        # calibration inputs
        self.defParams = torch.tensor([[1e3, -21e3]]) # standard presssure (MPa) and energy of ads. (J/mol)
        self.limits = [[5e2, 1.5e3],[-30e3, -15e3]] # range on defParams
                
        ## model constants
        self.RConst = torch.tensor(8.314) # universal gas constant (J/ mol/ K)
        self.data = None # dataset of model
        
        self.stdRatio = 0.1 # standard deviation ratio
        
        # TODO this isnt working: it needs to be solve_true when discrepamcy is true and solve_t when discrepancy is
        self.discrepancy = False
        if self.discrepancy:
            self.defOut = self.solve_t(self.defParams)
        else:
            self.defOut = self.solve_true(self.defParams)
       
        torch.manual_seed(35435)
        
    def solve_t(self, cal_inputs):

        num_batch = len(cal_inputs)
        num_vars = len(self.var_in)
        
        # unpack variable inputs
        T, P = torch.chunk(self.var_in, chunks = 2, dim = 1)
        T = T.repeat(1, num_batch)
        P = P.repeat(1, num_batch)

        # unpack calibration inputs
        p0Const, eConst = torch.chunk(cal_inputs, chunks = 2, dim = 1) # split cal_inputs into two chunks along the second dimension
        p0Const = p0Const.repeat(1, num_vars).t()
        eConst  = eConst.repeat(1, num_vars).t()
        
        # compute equilibrium constant
        kConst = 1/p0Const* torch.exp(-eConst / self.RConst / T)
        
        # compute surface coverage fraction
        cov_frac = kConst*P/(1 + kConst*P)

        # Return coverages        
        return cov_frac

    def solve_true(self, cal_inputs):

        num_batch = len(cal_inputs)
        num_vars = len(self.var_in)

        # unpack variable input
        T, P = torch.chunk(self.var_in, chunks = 2, dim = 1)
        T = T.repeat(1, num_batch)
        P = P.repeat(1, num_batch)

        # unpack ground truth calibration input (adsorption site 1)
        p01Const, e1Const = torch.chunk(cal_inputs, chunks = 2, dim = 1)
        p01Const = p01Const.repeat(1, num_vars).t()
        e1Const  = e1Const.repeat(1, num_vars).t()

        # specify adsorption site two parameters
        p02Const, e2Const, lambda1Const = torch.tensor(5000.0), torch.tensor(-22000.0), torch.tensor(0.5)
        p02Const = p02Const.repeat(1, num_vars).t()
        e2Const  = e2Const.repeat(1, num_vars).t()

        # compute equilibrium constant of site one
        k1Const = 1.0/p01Const * torch.exp(-e1Const / self.RConst / T)

        # compute equilibrium constant of site two
        k2Const = 1.0/p02Const * torch.exp(-e2Const / self.RConst / T)

        # compute surface coverage fraction for two adsorption sites with different equilibrium conditions
        cov_frac = lambda1Const * (k1Const*P/(1 + k1Const*P)) + (1 - lambda1Const) * (k2Const*P/(1 + k2Const*P))

        # Return
        return cov_frac
    
    def genDataFile(self, dataFileNamePrefix='observations', use_true_model=True, store = True, num_observations = 10):

        # solve model
        if(use_true_model):
            self.defOut = self.solve_true(self.defParams)
        else:
            self.defOut = self.solve_t(self.defParams)
        
        # get standard deviation
        stdDev = self.stdRatio * torch.abs(torch.mean(self.defOut))
        
        # add noise to coverage data
        coverage = self.defOut.repeat(1,num_observations)
        print('Coverage:', coverage)
        for loopA in range(num_observations):
            noise = torch.randn((len(coverage),1))*stdDev
            coverage[:,loopA] = coverage[:,loopA] + noise.flatten()
        
        if store:

            # specify file name
            file_name = f'{dataFileNamePrefix}.csv'

            with open(file_name, 'w') as file:

                # write file
                file.write('temperature, pressure, coverage \n')

                # specify row names
                for loopA in range(self.var_in.size(0)):
                    for loopB in range(self.var_in.size(1)):
                        file.write(f'{self.var_in[loopA,loopB]},')                        
                    for loopB in range(coverage.size(1)-1):
                        file.write(f'{coverage[loopA,loopB].item()},')
                    file.write(f'{coverage[loopA,coverage.size(1)-1].item()}\n')

# TEST SURROGATE
if __name__ == '__main__':

    torch.manual_seed(35435)

    var_grid = [[350.0],[1.0]]

    # Create model
    model = PhysChem(var_grid)
    
    # Generate data
    model.genDataFile(use_true_model = False, num_observations=1)


    