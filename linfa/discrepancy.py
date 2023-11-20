import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import path
from linfa.mlp import FNN

torch.set_default_tensor_type(torch.DoubleTensor)

class Discrepancy(object):
    """Class to create surrogate models for discrepancy estimatio
       TO BE COMPLETED!!!
    """
    def __init__(self, model_name,
                       lf_model,
                       input_size,
                       output_size,
                       var_grid_in,
                       var_grid_out,
                       dnn_arch=None, 
                       dnn_activation='relu', 
                       model_folder='./', 
                       surrogate=None, 
                       device='cpu'):
        
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = model_name
        self.model_folder = model_folder
        self.is_trained = False

        # Assign LF model
        self.lf_model = lf_model

        # Store variable grid locations
        self.var_grid_in=var_grid_in
        # Output variables - multiple noisy observations 
        # are available for each combination of variables
        self.var_grid_out=var_grid_out

        # Input/output statistics
        self.var_in_avg = torch.mean(var_grid_in,dim=0)
        if(len(self.var_grid_in) == 1):
            self.var_in_std = torch.zeros_like(self.var_in_avg)
        else:
            self.var_in_std = torch.std(var_grid_in,dim=0)
        # If there are multiple outputs, we will define one file for each output
        self.var_out_avg = torch.mean(var_grid_out)
        self.var_out_std = torch.std(var_grid_out)

        # Create surrogate
        self.surrogate = FNN(input_size, output_size, arch=dnn_arch, device=self.device, init_zero=True) if surrogate is None else surrogate
        
    def surrogate_save(self):
        """Save surrogate model to [self.name].sur and [self.name].npz
        
        Returns:
            None

        """
        # Save model state dictionary
        torch.save(self.surrogate.state_dict(), self.model_folder + self.model_name + '.sur')

    def surrogate_load(self):
        """Load surrogate model from [self.name].sur and [self.name].npz
        
        Returns:
            None
        """
        # Read back the state dictionary from file
        self.surrogate.load_state_dict(torch.load(self.model_folder + self.model_name + '.sur'))

    def update(self, batch_x, max_iters=10000, lr=0.01, lr_exp=0.999, record_interval=50, store=True, reg=False, reg_penalty=0.0001):
        """Train surrogate model with pre-grid.

        """
        print('')
        print('--- Training model discrepancy')
        print('')

        self.is_trained = True

        # LF model output at the current batch
        y = self.lf_model(batch_x) 

        # Standardize inputs and outputs
        if(len(self.var_grid_in) == 1):
            var_grid = torch.zeros_like(self.var_grid_in)
        else:
            var_grid = (self.var_grid_in - self.var_in_avg) / self.var_in_std
        # The output is the discrepancy
        var_out = self.var_grid_out - torch.mean(y,dim=1).unsqueeze(1)
        # Update output stats
        # self.var_out_avg = torch.mean(var_out)
        # self.var_out_std = torch.std(var_out)
        # Rescale outputs
        # var_out = (var_out - self.var_out_avg) / self.var_out_std

        # Set optimizer and scheduler
        optimizer = torch.optim.RMSprop(self.surrogate.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_exp)

        for i in range(max_iters):
            # Set surrogate in training mode
            self.surrogate.train()          
            # Surrogate returns a table with rows as batches and columns as variables considered            
            disc = self.surrogate(var_grid)

            # Compute loss averaged over batches/variables
            # Mean over the columns (batches)
            # Mean over the rows (variables)
            # Also we need to account for the number of repeated observations
            loss = torch.tensor(0.0)            
            # Loop over the number of observations
            for loopA in range(var_out.size(1)):
                loss += torch.sum((disc.flatten() - var_out[:,loopA]) ** 2)
                if reg:
                    reg_loss = 0
                    for param in self.surrogate.parameters():
                        reg_loss += torch.abs(param).sum() * reg_penalty
                    loss += reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % record_interval == 0:
                if reg:
                    print('DISC: it: %7d | loss: %8.3e | reg_loss: %8.3e' % (i, loss, reg_loss))
                else:
                    print('DISC: it: %7d | loss: %8.3e' % (i, loss))
        print('')
        print('--- Surrogate model pre-train complete')
        print('')        
        if store:
            self.surrogate_save()

    def forward(self, var):
        """Function to evaluate the surrogate
        
        Args:
            x (torch.Tensor): Contains a matrix of model inputs in the form [data_num, feature_dim]

        Returns:
            Value of the surrogate at x.

        """
        if(len(self.var_grid_in) == 1):
            res = self.surrogate(torch.zeros_like(var))
        else:
            res = self.surrogate((var - self.var_in_avg) / self.var_in_std)

        if not(self.is_trained):
            zero_res = torch.zeros_like(res)
            return zero_res
        else:
            return res

def test_surrogate():
    
    import matplotlib.pyplot as plt
    from linfa.models.discrepancy_models import PhysChem

    # Set variable grid
    var_grid = [[350.0, 400.0, 450.0],
                [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Create Model
    model = PhysChem(var_grid)

    # Generate true data
    model.genDataFile(dataFileNamePrefix='observations', use_true_model=True, store=True, num_observations=10)  

    # Get data from true model at the same TP conditions    
    var_data = np.loadtxt('observations.csv',skiprows=1,delimiter=',')
    var_data_in = torch.tensor(var_data[:,:2])
    var_data_out = torch.tensor(var_data[:,2:])
    
    # Define emulator and pre-train on global grid
    discrepancy = Discrepancy(model_name='discrepancy_test',
                              lf_model=model.solve_lf,
                              input_size=2,
                              output_size=1,
                              var_grid_in=var_data_in,
                              var_grid_out=var_data_out)
    
    # Create a batch of samples for the calibration parameters
    batch_x = model.defParams

    # Update the discrepancy model
    discrepancy.update(batch_x, max_iters=1000, lr=0.001, lr_exp=0.9999, record_interval=100)

    # Plot discrepancy
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(model.var_in[:,0].detach().numpy(), model.var_in[:,1].detach().numpy(), discrepancy.forward(model.var_in).detach().numpy(), marker='o')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Pressure')
    ax.set_zlabel('Coverage')
    plt.show()

# TEST SURROGATE
if __name__ == '__main__':
  
  test_surrogate()
    