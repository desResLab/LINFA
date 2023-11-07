#NOTES
#did not change any of the NOFAS parameters to work with multifidelity
#left optimizer and learning rate scheduler the same for both model optimizations, could be adjusted
#did not change annealing parameters for multifidelity since the idea is we will only apply annealing to low fidelity model

import os
import torch
import numpy as np
from linfa.maf import MAF, RealNVP

torch.set_default_tensor_type(torch.DoubleTensor)

class experiment_mf:
    """Defines an instance of variational inference

    This class is the core class of the LINFA library
    and defines all the default hyperparameter values and 
    and functions used for inference. 
    """
    def __init__(self):

        # NF ARCHITECTURE parameters
        self.name              = "Experiment"
        self.input_size        = 3             
        self.flow_type         = 'maf'         
        self.n_blocks_low      = 15     #change n_blocks to n_blocks_low for the low fidelity model
        self.n_blocks_high     = 20     #add n_blocks_high for the high fidelity model (must be greater than n_blocks_low)
        self.hidden_size       = 100           
        self.n_hidden          = 1             
        self.activation_fn     = 'relu'        
        self.input_order       = 'sequential'  
        self.batch_norm_order  = True          

        # NOFAS parameters
        self.run_nofas          = True  
        self.log_interval       = 10    
        self.calibrate_interval = 300   
        self.true_data_num      = 2     
        self.budget             = 216   
        self.surr_pre_it        = 40000 
        self.surr_upd_it        = 6000  
        self.surr_folder        = "./"  
        self.use_new_surr       = True  

        # OPTIMIZER parameters
        self.optimizer       = 'Adam'   
        self.lr_low          = 0.003   #change lr to lr_low for the low fidelity model optimization
        self.lr_high         = 0.003   #add lr_high for the high fidelity model optimization
        self.lr_decay_low    = 0.9999  #change lr_decay to lr_decay_low for the low fidelity model optimization
        self.lr_decay_high   = 0.9999  #add lr_decay_high for the high fidelity model optimization
        self.lr_scheduler    = 'StepLR' 
        self.lr_step_low     = 1000    #change lr_step to lr_step_low for the low fidelity model optimization
        self.lr_step_high    = 1000    #add lr_step_high for the high fidelity model optimization  
        self.batch_size_low  = 500     #change batch_size to batch_size_low for the low fidelity model optimization
        self.batch_size_high = 500     #add batch_size_high for the high fidelity model optimization        
        self.n_iter_low      = 25001   #change n_iter to n_iter_low for the low fidelity model optimization
        self.n_iter_high     = 25001   #add n_iter_high for the high fidelity model optimization

        # ANNEALING parameters
        self.annealing     = True     
        self.scheduler     = 'AdaAnn' 
        # AdaAnn
        self.tol           = 0.001    
        self.t0            = 0.01     
        self.N             = 100      
        self.N_1           = 1000     
        self.T_0           = 500      
        self.T             = 5        
        self.T_1           = 5001     
        self.M             = 1000             
        # Linear scheduler
        self.linear_step  = 0.0001   

        # OUTPUT parameters
        self.output_dir          = './results/' + self.name 
        self.log_file            = 'log.txt'                
        self.seed                = 35435                    
        self.n_sample            = 5000                     
        self.save_interval       = 200                      
        self.store_nf_interval   = 1000                     
        self.store_surr_interval = None                     

        # DEVICE parameters
        self.no_cuda = True #:bool: Flag to use CPU

        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not self.no_cuda else 'cpu')

        # Local pointer to the main components for inference
        self.transform             = None
        self.model_low             = None #change model to model_low for low fidelity model
        self.model_high            = None #add model_high for high fidelity model
        self.model_logdensity_low  = None #change model_logdensity to model_logdensity_low for low fidelity model
        self.model_logdensity_high = None #add model_logdensity_high for high fidelity model
        self.surrogate             = None

        
    def run(self):
        """Runs instance of inference problem        

        """

        # Check is surrogate exists
        if self.run_nofas:
            if not os.path.exists(self.name + ".sur") or not os.path.exists(self.name + ".npz"):
                print("Abort: NoFAS enabled, without surrogate files. \nPlease include the following surrogate files in root directory.\n{}.sur and {}.npz".format(self.name, self.name))
                exit(0)
        # setup file ops
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Save a copy of the data in the result folder so it is handy
        if hasattr(self.model_low,'data'):
          np.savetxt(self.output_dir + '/' + self.name + '_data', self.model.data, newline="\n") #change to save low fidelity data
        if hasattr(self.model_high,'data'):
          np.savetxt(self.output_dir + '/' + self.name + '_data', self.model.data, newline="\n") #add save high fidelity data

        # setup device
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda': torch.cuda.manual_seed(self.seed)

        print('')
        print('--- Running on device: '+ str(self.device))
        print('')

        #CHANGE: set up the first normalizing flow for the low fidelity model optimization here
        # the only change here is the n_blocks_low so the number of layers differ between nf_low and nf_high
        # the remaining parameters need to be the same so the nf_low and nf_high have the same structure
        
        # model
        if self.flow_type == 'maf':
            nf_low = MAF(self.n_blocks_low, self.input_size, self.hidden_size, self.n_hidden, None,
                     self.activation_fn, self.input_order, batch_norm=self.batch_norm_order)
        elif self.flow_type == 'realnvp':  # Under construction
            nf_low = RealNVP(self.n_blocks_low, self.input_size, self.hidden_size, self.n_hidden, None,
                         batch_norm=self.batch_norm_order)
        else:
            raise ValueError('Unrecognized model.')

        
        #CHANGE: set the optimizer and learning rate scheduler for the low fidelity model
        # set nf to nf_low
        # set lr to lr_low
        # set lr_step to lr_step_low
        # set lr_decay to lr_decay_low
        
        nf_low = nf_low.to(self.device)
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(nf_low.parameters(), lr=self.lr_low)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(nf_low.parameters(), lr=self.lr_low)
        else:
            raise ValueError('Unrecognized optimizer.')

        if self.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_step_low, self.lr_decay_low)
        elif self.lr_scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay_low)
        else:
            raise ValueError('Unrecognized learning rate scheduler.')

        if self.annealing:
            i = 1
            prev_i = 1
            tvals = []
            t = self.t0
            dt = 0.0
            loglist = []
            while t < 1:
                t = min(1, t + dt)
                tvals = np.concatenate([tvals, np.array([t])])
                self.n_iter_low = self.T
                self.batch_size_low = self.N
                
                if t == self.t0:
                    self.n_iter_low = self.T_0
                if t == 1:
                    self.batch_size_low = self.N_1
                    self.n_iter_low = self.T_1

                while i < prev_i + self.n_iter_low:
                    self.train(nf_low, optimizer, i, loglist, low_fidelity=True, sampling=True, t=t) #added low_fidelity flag
                    if t == 1:
                        scheduler.step()
                    i += 1
                prev_i = i

                if self.scheduler == 'AdaAnn':
                    z0 = nf_low.base_dist.sample([self.M])
                    zk, _ = nf_low(z0)
                    log_qk = self.model_logdensity_low(zk)
                    dt = self.tol / torch.sqrt(log_qk.var())
                    dt = dt.detach()# .numpy()

                if self.scheduler == 'Linear':
                    dt = self.linear_step
        else:
            loglist = []
            for i in range(1, self.n_iter_low+1):                
                self.train(nf_low, optimizer, i, loglist, low_fidelity=True, sampling=True)
                scheduler.step()

                
        #ADD: set up the second normalizing flow for the high fidelity model optimization here
        # the only change here is the n_blocks_high
        
        # model
        if self.flow_type == 'maf':
            nf_high = MAF(self.n_blocks_high, self.input_size, self.hidden_size, self.n_hidden, None,
                     self.activation_fn, self.input_order, batch_norm=self.batch_norm_order)
        elif self.flow_type == 'realnvp':  # Under construction
            nf_high = RealNVP(self.n_blocks_high, self.input_size, self.hidden_size, self.n_hidden, None,
                         batch_norm=self.batch_norm_order)
        else:
            raise ValueError('Unrecognized model.')

        
        #CHANGE: set the optimizer and learning rate scheduler for the high fidelity model
        # reinitializing the optimizer and scheduler, we can rename instead
        # set nf to nf_high
        # set lr to lr_high
        # set lr_step to lr_step_high
        # set lr_decay to lr_decay_high
        
        nf_high = nf_high.to(self.device)
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(nf_high.parameters(), lr=self.lr_high)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(nf_high.parameters(), lr=self.lr_high)
        else:
            raise ValueError('Unrecognized optimizer.')

        if self.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_step_high, self.lr_decay_high)
        elif self.lr_scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay_high)
        else:
            raise ValueError('Unrecognized learning rate scheduler.')

        
        #CHANGE: compute the statistics for the batch norm layers
                
        x00 = nf_low.base_dist.sample([10000])
        xkk, _ = nf_low(x00)
        
        mean_vals = xkk.mean(dim=0)
        log_std_vals = torch.log(xkk.std(dim=0)) #log since the batch norm uses log of standard deviation
        
        
        #CHANGE: update the state dictionary for the high fidelity model with the parameters from the low fidelity model
        
        state_dict = nf_high.state_dict()
        state_dict.update(nf_low.state_dict())
        
        
        #CHANGE: update the remaining batch norm layers in the high fidelity model using the computed statistics above
        # the batch norm layers fall on the odd layers of the realnvp framework
        # this is only currently working for realnvp, I still need to look into the maf framework
        
        for j in range(2*self.n_blocks_low + 1, 2*self.n_blocks_high):
        
            if j % 2 == 1:
                state_dict['net.' + str(j) + '.log_gamma'] = log_std_vals
                state_dict['net.' + str(j) + '.beta'] = mean_vals

        
        #CHANGE: update the high fideltiy model with the updated parameters
        nf_high.load_state_dict(state_dict)
        
        
        #CHANGE: determine the length of the parameter list in low fidelity model
        length = len(torch.nn.ParameterList(nf_low.parameters()))
        
        
        #CHANGE: freeze the layers corresponding to the low fidelity model in the new high fidelity model
        # do this by changing the requires_grad parameter to False 
        
        j = 0
        for param in nf_high.parameters():
            param.requires_grad = False
            j+=1
            if j == length:
                break
                
        
        #CHANGE: optimize the high fidelity model
        # did not include annealing here since we are optimizing the high fidelity model, we can always change this
        
        #loglist = []
        for i in range(1, self.n_iter_high+1):                
            self.train(nf_high, optimizer, i, loglist, low_fidelity=False, sampling=True)
            scheduler.step()
                
        print('')
        print('--- Simulation completed!')        

        
    def train(self, nf, optimizer, iteration, log, low_fidelity, sampling=True, t=1): #added a flag to switch between low and high fidelity models
        """Parameter update for normalizing flow and surrogate model

        This is the function where the ELBO loss function is evaluated, 
        the results are saved and the surrogate model is updated.
        
        Args:
            nf (instance of normalizing flow): the normalizing flow architecture used for variational inference
            optimizer (instance of PyTorch optimizer): the selected PyTorch optimizer
            iteration (int): current iteration number
            log (list of lists): stores a log of [iteration, annealing temperature, loss value]
            sampling (bool): Flag indicating the sampling stage
            t (double): current inverse temperature for annealing scheduler

        Returns:
            None
        """

        # Set the normalizing flow in training mode
        nf.train()

        # Evaluate the Jacobian terms in  loss function
        
        if low_fidelity == True:
            x0 = nf.base_dist.sample([self.batch_size_low]) #low fidelity model batch size
            
        else:
            x0 = nf.base_dist.sample([self.batch_size_high]) #high fidelity model batch size
        
        xk, sum_log_abs_det_jacobians = nf(x0)

        # generate and save samples evaluation
        if sampling and iteration % self.save_interval == 0:
            print('--- Saving results at iteration '+str(iteration))
            x00 = nf.base_dist.sample([self.n_sample])
            xkk, _ = nf(x00)
            # Save surrogate grid
            if not(self.surrogate is None):
              np.savetxt(self.output_dir + '/' + self.name + '_grid_' + str(iteration), self.surrogate.grid_record.clone().cpu().numpy(), newline="\n")
            # Save log profile
            np.savetxt(self.output_dir + '/' + self.log_file, np.array(log), newline="\n")
            # Save transformed samples          
            np.savetxt(self.output_dir + '/' + self.name + '_samples_' + str(iteration), xkk.data.clone().cpu().numpy(), newline="\n")
            # Save samples in the original space
            if not(self.transform is None):
              xkk_samples = self.transform.forward(xkk).data.cpu().numpy()
              np.savetxt(self.output_dir + '/' + self.name + '_params_' + str(iteration), xkk_samples, newline="\n")
            else:
              xkk_samples = xkk.data.cpu().numpy()
              np.savetxt(self.output_dir + '/' + self.name + '_params_' + str(iteration), xkk_samples, newline="\n")
            # Save marginal statistics
            np.savetxt(self.output_dir + '/' + self.name + '_marginal_stats_' + str(iteration), np.concatenate((xkk_samples.mean(axis=0).reshape(-1,1),xkk_samples.std(axis=0).reshape(-1,1)),axis=1), newline="\n")
            # Save log density at the same samples
            np.savetxt(self.output_dir + '/' + self.name + '_logdensity_LF_' + str(iteration), self.model_logdensity_low(xkk).data.cpu().numpy(), newline="\n")
            # Save model outputs at the samples - If a model is defined
            if not(self.transform is None):
              stds = torch.abs(self.model.defOut).to(self.device) * self.model.stdRatio
              o00 = torch.randn(x00.size(0), self.model.data.shape[0]).to(self.device)
              noise = o00*stds.repeat(o00.size(0),1)
              if self.surrogate:
                np.savetxt(self.output_dir + '/' + self.name + '_outputs_' + str(iteration), (self.surrogate.forward(xkk) + noise).data.cpu().numpy(), newline="\n")
              else:
                np.savetxt(self.output_dir + '/' + self.name + '_outputs_' + str(iteration), (self.model.solve_t(self.transform.forward(xkk)) + noise).data.cpu().numpy(), newline="\n")

        if torch.any(torch.isnan(xk)):
            print("Error: samples xk are nan at iteration " + str(iteration))
            print(xk)
            np.savetxt(self.output_dir + '/' + self.log_file, np.array(log), newline="\n")
            exit(-1)

        # updating surrogate model
        if self.run_nofas and iteration % self.calibrate_interval == 0 and self.surrogate.grid_record.size(0) < self.budget:
            xk0 = xk[:self.true_data_num, :].data.clone()            
            # print("\n")
            # print(list(self.surrogate.grid_record.size())[0])
            # print(xk0)
            self.surrogate.update(xk0, max_iters=self.surr_upd_it)

        # Free energy bound
        
        
        #CHANGE: need to switch between the low and high fidelity models in the loss function
        # added a flag in the training function depending on which model is being optimized
        # true for low fidelity model, false for high fidelity model
        
        if low_fidelity == True:
            loss = (- torch.sum(sum_log_abs_det_jacobians, 1) - t * self.model_logdensity_low(xk)).mean()
            
        else:
            loss = (- torch.sum(sum_log_abs_det_jacobians, 1) - t * self.model_logdensity_high(xk)).mean()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % self.log_interval == 0:
            print('VI NF (t=%5.3f): it: %7d | loss: %8.3e' % (t,iteration, loss.item()))
            log.append([t, iteration, loss.item()])
        
        # Save state of normalizing flow layers
        if self.store_nf_interval > 0 and iteration % self.store_nf_interval == 0:
            torch.save(nf.state_dict(), self.output_dir + '/' + self.name + "_" + str(iteration) + ".nf")

        if not(self.store_surr_interval is None) and self.store_surr_interval > 0 and iteration % self.store_surr_interval == 0:
            self.surrogate.surrogate_save() # Save surrogate model        

