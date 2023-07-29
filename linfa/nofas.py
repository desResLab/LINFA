import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import path

torch.set_default_tensor_type(torch.DoubleTensor)

class FNN(nn.Module):
    """Fully Connected Neural Network"""

    def __init__(self, input_size, output_size, arch=None, activation='relu', device='cpu'):
        """
        Args:
            input_size (int): Input size for FNN
            output_size (int): Output size for FNN
            arch (list of int): list containing the number of neurons for each hidden layer
            activation (stirng): type of activation function, either 'relu', 
            device (string): computational device
        """
        super().__init__()
        if(arch is None):
            self.neuron_size = [64,32]
        else:
            self.neuron_size = arch
        # Set activation
        self.activation = activation
        self.fc = nn.ModuleList()
        # Assign first layer
        self.fc.append(nn.Linear(input_size, self.neuron_size[0]).to(device))
        if(len(self.neuron_size) == 1):
            # Assign output Layer
            self.fc.append(nn.Linear(self.neuron_size[0], output_size).to(device))
        else:
            for layer in range(1,len(self.neuron_size)):
                self.fc.append(nn.Linear(self.neuron_size[layer-1], self.neuron_size[layer]).to(device))
            # Assign output layer
            self.fc.append(nn.Linear(self.neuron_size[len(self.neuron_size)-1], output_size).to(device))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [num_samples, input_size], assumed to be a batch.

        Returns:
            torch.Tensor. Assumed to be a batch.
        """
        for loopA in range(len(self.fc)-1):
            if(self.activation == 'relu'):
                x = F.relu(self.fc[loopA](x))
            elif(self.activation == 'silu'):
                x = F.silu(self.fc[loopA](x))
            elif(self.activation == 'tanh'):
                x = F.tanh(self.fc[loopA](x))
            else:
                print('Invalid activation string.')
                exit(-1)

        # Last layer with linear activation
        x = self.fc[len(self.fc)-1](x)
        return x

class Surrogate:
    """Class to create surrogate models for inference with NoFAS

       Args:
            model_name (string): name of the true model to be approximated
            model_func (function): with only one input of torch.Tensor
            input_size (int): input dimension of true model.
            output_size (int): output dimension of true model.
            dnn_arch (list of int): list containing the number of neurons for each hidden layer. Default None for [64,32] architecture.
            dnn_activation (stirng): type of activation function, either 'relu', 
            model_folder (string): folder where surrogate model is stored.             
            limits (list or lists): bounds for all inputs, in format of [[low_0, high_0], [low_1, high_1], ... ]
            memory_len (int): the maximal number of batches stored in buffer. Default: 20
            surrogate (None or torch.nn.Module): the implementation of surrogate model used. Default: FNN This allows the user to spacify any NN architecture for the surrogate.
                                                 If None the parameters "dnn_arch" and "dnn_activation" can be used to specify a dense neural network. 
    """
    def __init__(self, model_name, model_func, input_size, output_size, dnn_arch=None, dnn_activation='relu', model_folder='./', limits=None, memory_len=20, surrogate=None, device='cpu'):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = model_name
        self.model_folder = model_folder
        self.mf = model_func
        self.pre_out = None
        self.m = None
        self.sd = None
        self.tm = None
        self.tsd = None
        self.limits = limits
        self.pre_grid = None
        self.surrogate = FNN(input_size, output_size, arch=dnn_arch, device=self.device) if surrogate is None else surrogate
        self.beta_0 = 0.5
        self.beta_1 = 0.1        

        self.memory_grid = []
        self.memory_out = []
        self.memory_len = memory_len
        self.weights = torch.Tensor([np.exp(-self.beta_1 * i) for i in range(memory_len)]).to(self.device)
        self.grid_record = None
        

    @property
    def limits(self):
        """Retrieves the limits of all inputs
        """
        return self.__limits

    @limits.setter
    def limits(self, limits):
        """Sets the limits of all inputs
        
        Args:
            limits (list or tuple): Lists lower and upper bound in the format *[[low_0, high_0], [low_1, high_1], ...]*

        """
        limits = torch.Tensor(limits).tolist()
        if len(limits) != self.input_size:
            print("Error: Invalid input size for limit. Abort.")
            exit(-1)
        elif any(len(item) != 2 for item in limits):
            print("Error: Limits should be two bounds. Abort.")
            exit(-1)
        elif any(item[0] > item[1] for item in limits):
            print("Error: Upper bound should not be smaller than lower bound. Abort.")
            exit(-1)
        self.__limits = limits

    @property
    def pre_grid(self):
        return self.__pre_grid

    @pre_grid.setter
    def pre_grid(self, pre_grid):
        """Assign the pregrid for the surrogate model. 

        f nofas is not enabled, this serves as the whole training grid.
        
        Args:
            pre_grid (None or torch.Tensor): Specifies a matrix of model inputs with size [data_num, feature_dim]. If None, will try to search for *npz* containing this info.

        Returns:
            None

        """
        if pre_grid is None:
            if path.exists(self.model_folder + self.model_name + '.npz'):
                container = np.load(self.model_folder + self.model_name + '.npz')
                if 'pre_grid' in container:
                    self.pre_grid = container['pre_grid']
                    print("Success: Pre-Grid found.")
            else:
                print("Warning: " + self.model_folder + self.model_name + ".npz does not found, please generate pre-grid.")
                print("Suggestion: Use Surrogate.gen_grid(input_limits=None, grid_num=5, store=True)")
        else:
            self.__pre_grid = torch.Tensor(pre_grid).to(self.device)
            self.m = torch.mean(self.pre_grid, 0)
            self.sd = torch.std(self.pre_grid, 0)
            # Evaluate model at pre-grid
            self.pre_out = self.mf(self.pre_grid)
            self.tm = torch.mean(self.pre_out, 0)
            self.tsd = torch.std(self.pre_out, 0)
            self.grid_record = self.__pre_grid.clone()

    def gen_grid(self, input_limits=None, grid_type='tensor', gridnum=4, store=True):
        """Generates a pre-grid.
        
        Args:
            input_limits (None or list[list]): If None, use self.limits. If list[list], rewrite self.limits and use it.
            gridnum (int): Contains the number of grid points per dimension.
            store (bool): Flag indicating the pre-grid is store in self.pre_grid. If False then self.pre_grid is None.

        Returns:
            torch.Tensor: Input values for the full tensor grid in *dim* dimensions stored in a matrix [gridnum ** dim, dim].
        """
        if (grid_type == 'tensor'):
          
            meshpoints = []
            if input_limits is not None:
                self.limits = input_limits
                print("Warning: Input limits recorded in surrogate.")

            for lim in self.limits: meshpoints.append(torch.linspace(lim[0], lim[1], steps=gridnum))
            grid = torch.meshgrid(meshpoints,indexing='ij')
            grid = torch.cat([item.reshape(gridnum ** len(self.limits), 1) for item in grid], 1)

        elif (grid_type == 'sobol'):

            # Generate sobol samples in [0,1]^d
            soboleng = torch.quasirandom.SobolEngine(dimension=len(self.limits))
            grid = soboleng.draw(gridnum)
            for i,lim in enumerate(self.limits): 
                grid[:,i] = lim[0] + (lim[1] - lim[0]) * grid[:,i]

        else:

            print('Invalid type for pre-grid generation')
            exit(-1)

        # Store pre-grid
        if store:
            self.pre_grid = grid
            self.grid_record = self.pre_grid.clone()
            self.surrogate_save()
        
        # Return grid
        return grid

    def surrogate_save(self):
        """Save surrogate model to [self.name].sur and [self.name].npz
        
        Returns:
            None

        """
        torch.save(self.surrogate.state_dict(), self.model_folder + self.model_name + '.sur')
        np.savez(self.model_folder + self.model_name, limits=self.limits, pre_grid=self.pre_grid.clone().cpu().numpy(),
                 grid_record=self.grid_record.clone().cpu().numpy())

    def surrogate_load(self):
        """Load surrogate model from [self.name].sur and [self.name].npz
        
        Returns:
            None
        """
        self.surrogate.load_state_dict(torch.load(self.model_folder + self.model_name + '.sur'))
        container = np.load(self.model_folder + self.model_name + '.npz')
        for key in container:
            try:
                setattr(self, key, torch.Tensor(container[key]))
                print("Success: [" + key + "] loaded.")
            except:
                print("Warning: [" + key + "] is not a surrogate variables.")

    def pre_train(self, max_iters, lr, lr_exp, record_interval, store=True, reg=False):
        """Train surrogate model with pre-grid.

         Training is performed with the RMSprop optimizer and with an exponential learning rate scheduler.

        Args:
            max_iters (int): The maximal number of iterations.
            lr (double): Learning rate for RMSprop.
            lr_exp (double): Decay factor for exponential scheduler.
            record_interval (int): The number of iterations to print loss info
            store (bool): If true, self.surrogate_save() will be called.
            reg (bool): If true, L1 regularization will be used with parameter 0.0001

        Returns:
            None
        """
        print('')
        print('--- Pre-training surrogate model')
        print('')
        grid = (self.pre_grid - self.m) / self.sd
        out = (self.pre_out - self.tm) / self.tsd
        optimizer = torch.optim.RMSprop(self.surrogate.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_exp)
        for i in range(max_iters):
            self.surrogate.train()            
            y = self.surrogate(grid)
            loss = torch.sum((y - out) ** 2) / y.size(0)
            if reg:
                reg_loss = 0
                for param in self.surrogate.parameters():
                    reg_loss += torch.abs(param).sum() * 0.0001
                loss += reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % record_interval == 0:
                if reg:
                    print('SUR: PRE: it: %7d | loss: %8.3e | reg_loss: %8.3e' % (i, loss, reg_loss))
                else:
                    print('SUR: PRE: it: %7d | loss: %8.3e' % (i, loss))
        print('')
        print('--- Surrogate model pre-train complete')
        print('')
        if store:
            self.surrogate_save()

    def update(self, x, max_iters=10000, lr=0.01, lr_exp=0.999, record_interval=500, store=False, tol=1e-5, reg=False):
        """Model calibration for NoFAS.
        
        Fine tuning is performed with the RMSprop optimizer and an exponential learning rate scheduler.
        
        Args:
            x (torch.Tensor): Input feature that needs true model output.
            max_iters (int): Maximum number of iteration. Default 10000.
            lr (double): Learning rate for RMSprop. Default 0.01.
            lr_exp (double): Decay factor of exponential scheduler.
            record_interval (int): The number of iterations to print loss information.
            store (bool): If ture, self.surrogate_save() will be called.
            tol (double): Optimization will be terminated if loss < tol ** 2.
            reg (bool): If ture, L1 regularizaiton will be used with parameter 0.1.

        Returns:
            None
        """
        self.grid_record = torch.cat((self.grid_record.to(self.device), x), dim=0)
        s = torch.std(x, dim=0)
        thresh = torch.tensor(0.1).to(self.device)
        if torch.any(s < thresh):
            p = x[:, s < thresh]
            x[:, s < thresh] += torch.normal(0, 1, size=tuple(p.size())).to(self.device) * thresh
        s_aft = torch.std(x, dim=0)            
        
        # Print the std for each dimension before and after inflation    
        print('')
        print('--- Updating surrogate model')
        print('')
        print('Std before inflation -> Std after inflation')
        for loopA in range(s.size(0)):
          print("%8.3e -> %8.3e" % (s[loopA],s_aft[loopA]))
        print('')          

        if len(self.memory_grid) >= self.memory_len:
            self.memory_grid.pop()
            self.memory_out.pop()
        self.memory_grid.insert(0, (x - self.m) / self.sd)
        self.memory_out.insert(0, (self.mf(x) - self.tm) / self.tsd)
        sizes = [list(self.pre_grid.size())[0]] + [list(item.size())[0] for item in self.memory_grid]

        optimizer = torch.optim.RMSprop(self.surrogate.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_exp)
        for i in range(max_iters):
            self.surrogate.train()            
            y = self.surrogate(torch.cat(((self.pre_grid - self.m) / self.sd, *self.memory_grid), dim=0))
            out = torch.cat(((self.pre_out - self.tm) / self.tsd, *self.memory_out), dim=0)
            raw_loss = torch.stack([item.mean() for item in torch.split(torch.sum((y - out) ** 2, dim=1), sizes)])
            loss = raw_loss[0] * 2 * self.beta_0 * self.weights[:len(self.memory_grid)].sum() + \
                   torch.sum(raw_loss[1:] * self.weights[:len(self.memory_grid)]) * (1 - self.beta_0) * 2

            # loss = raw_loss[0] * self.weights[:len(self.memory_grid)].sum() + torch.sum(
            #     raw_loss[1:] * self.weights[:len(self.memory_grid)])

            if reg:
                for param in self.surrogate.parameters():
                    loss += torch.abs(param).sum() * 0.1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % record_interval == 0:
                # print('Updating: {}\t loss {}'.format(i, loss), end='\r')
                print('SUR: UPD: it: %7d | loss: %8.3e' % (i, loss))
            if loss < tol ** 2: break
        # print('                                                        ', end='\r')
        print('')
        print('--- Surrogate model updated')
        print('')                  
        if store:
            self.surrogate_save()

    def forward(self, x):
        """Function to evaluate the surrogate
        
        Args:
            x (torch.Tensor): Contains a matrix of model inputs in the form [data_num, feature_dim]

        Returns:
            Value of the surrogate at x.

        """
        return self.surrogate((x - self.m) / self.sd) * self.tsd + self.tm

def test_surrogate():
    
    import matplotlib.pyplot as plt
  
    # Define function to approximate
    def ishigami(z):
        z1, z2, z3 = torch.chunk(z, chunks=3, dim=1)
        return torch.sin(z1) + 7.0 * (torch.sin(z2))**2 + 0.1 * z3**4 * torch.sin(z1)
    
    # Set surrogate architecture
    # arch=[64,64,64,64]
    arch=None
    activation='silu'
    
    # Define emulator and pre-train on global grid
    curr_limits = [[-torch.pi,torch.pi],[-torch.pi,torch.pi],[-torch.pi,torch.pi]]
    emulator    = Surrogate(model_name='ishigami', model_func=ishigami, input_size=3, output_size=1, limits = curr_limits, 
                            dnn_arch=arch, dnn_activation=activation)
    emulator.gen_grid(gridnum=10)
    emulator.pre_train(40000, 0.03, 0.9999, 500, store=True)

    # Eval MSE on Sobol point collection
    soboleng = torch.quasirandom.SobolEngine(dimension=3)
    # 2**10 = 1024 testing points
    z = (soboleng.draw_base2(10) - 0.5) * 2 * torch.pi
    true_fc = ishigami(z)
    emul_fc = emulator.forward(z)
    mse = torch.sum((true_fc - emul_fc) ** 2) / z.size(0)
    print('Test MSE: %.3f' % (mse.item()))

    # Check loss on pregrid
    true_fc_pg = ishigami(emulator.pre_grid)
    emul_fc_pg = emulator.forward(emulator.pre_grid)
    mse = torch.sum((true_fc_pg - emul_fc_pg) ** 2) / emulator.pre_grid.size(0)
    print('Pre-grid MSE: %.3f' % (mse.item()))
    print('Standard deviation on testing set: %.3f' % (emulator.tsd.item()))

        # Plot output histograms
    plt.figure(figsize=(5,5))
    plt.hist(true_fc.detach().numpy(),bins='auto',alpha=0.5,density=True)
    plt.hist(emul_fc.detach().numpy(),bins='auto',alpha=0.5,density=True)
    plt.show()

# TEST SURROGATE
if __name__ == '__main__':
  
  test_surrogate()
    