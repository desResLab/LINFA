import os
import torch
import numpy as np
from linfa.maf import MAF, RealNVP

torch.set_default_tensor_type(torch.DoubleTensor)

class experiment:
    """Defines an instance of variational inference

    This class is the core class of the LINFA library
    and defines all the default hyperparameter values and 
    and functions used for inference. 
    """
    def __init__(self):

        # NF ARCHITECTURE parameters
        self.name              = 'Experiment'
        self.input_size        = 3             #:int:  Number of input parameters
        self.flow_type         = 'maf'         #:str:  Type of flow ('maf' or 'realnvp')
        self.n_blocks          = 15            #:int:  Number of layers
        self.hidden_size       = 100           #:int:  Hidden layer size for MADE in each layer
        self.n_hidden          = 1             #:int:  Number of hidden layers in each MADE
        self.activation_fn     = 'relu'        #:str:  Actication function used (either 'relu','tanh' or 'sigmoid')
        self.input_order       = 'sequential'  #:str:  Input order for create_mask (either 'sequential' or 'random')
        self.batch_norm_order  = True          #:bool: Uses decide if batch_norm is used

        # NOFAS parameters
        self.run_nofas          = True        #:bool:   Activate NoFAS and the use of a surrogate model
        self.surrogate_type     = 'surrogate' #:str:    Type of surrogate model ('surrogate' or 'discrepancy')
        self.log_interval       = 10          #:int:    How often the loss statistics are printed
        self.calibrate_interval = 300         #:int:    How often the surrogate model is updated
        self.true_data_num      = 2           #:double: Number of true model evaluated at each surrogate update
        self.budget             = 216         #:int:    Maximum number of allowed evaluations of the true model
        self.surr_pre_it        = 40000       #:int:    Number of pre-training iterations for surrogate model
        self.surr_upd_it        = 6000        #:int:    Number of iterations for the surrogate model update
        self.surr_folder        = "./"        #:str:    Folder where the surrogate model is stored
        self.use_new_surr       = True        #:bool:   Start by pre-training a new surrogate and ignore existing surrogates

        # OPTIMIZER parameters
        self.optimizer    = 'Adam'   #:str:    Type of optimizer used (either 'Adam' or 'RMSprop')
        self.lr           = 0.003    #:double: Learning rate
        self.lr_decay     = 0.9999   #:double: Learning rate decay
        self.lr_scheduler = 'StepLR' #:str:    type of lr scheduler used (either 'StepLR' or 'ExponentialLR')
        self.lr_step      = 1000     #:int:    Number of steps for StepLR learning rate scheduler 
        self.batch_size   = 500      #:int:    Number of batch samples generated at every iteration from the base distribution        
        self.n_iter       = 25001    #:int:    Total number of iterations

        # ANNEALING parameters
        self.annealing    = True     #:bool:   Flag to activate an annealing scheduler
        self.scheduler    = 'AdaAnn' #:str:    Type of annealing scheduler (either 'AdaAnn' or 'Linear')
        # AdaAnn
        self.tol          = 0.001    #:double: KL tolerance for AdaAnn scheduler
        self.t0           = 0.01     #:double: Initial value for the inverse temperature
        self.N            = 100      #:int:    Number of batch samples generated for $t<1$ at each iteration
        self.N_1          = 1000     #:int:    number of batch samples generated for $t=1$ at each iteration
        self.T_0          = 500      #:int:    Number of parameter updates at the initial inverse temperature $t_0$
        self.T            = 5        #:int:    Number of parameter updates for each temperature for $t<1$
        self.T_1          = 5001     #:int:    Number of parameter updates at $t=1$
        self.M            = 1000     #:int:    Number of Monte Carlo  samples use to compute the denominator of the AdaAnn formula        
        # Linear scheduler
        self.linear_step  = 0.0001   #:double: Fixed step size for the Linear annealing scheduler

        # OUTPUT parameters
        self.output_dir          = './results/' + self.name #:str: Name of the output folder
        self.log_file            = 'log.txt'                #:str: File name where the log profile stats are written
        self.seed                = 35435                    #:int: Random seed
        self.n_sample            = 5000                     #:int: Number of batch samples used to print results at save_interval
        self.save_interval       = 200                      #:int: Save interval for all results
        self.store_nf_interval   = 1000                     #:int: Save interval for normalizing flow parameters
        self.store_surr_interval = None                     #:int: Save interval for surrogate model (None for no save)

        # DEVICE parameters
        self.no_cuda = True #:bool: Flag to use CPU

        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not self.no_cuda else 'cpu')

        # Local pointer to the main components for inference
        self.transform        = None
        self.model            = None
        self.model_logdensity = None
        self.surrogate        = None
        self.model_logprior   = None

    def run(self):
        """Runs instance of inference inference problem        

        """

        # Check is surrogate exists
        if self.run_nofas:
            if (self.surrogate_type == 'surrogate'):
                not_found = not os.path.exists(self.name + ".sur") or not os.path.exists(self.name + ".npz")
            elif(self.surrogate_type == 'discrepancy'):
                # !!! Temprary - CHECK!!!                
                not_found = False
                # not_found = not os.path.exists(self.name + ".sur")
            else:
                print('Invalid type of surrogate model')
                exit(-1)
            if(not_found):
                print("Abort: NoFAS enabled, without surrogate files. \nPlease include the following surrogate files in root directory.\n{}.sur and {}.npz".format(self.name, self.name))
                exit(-1)

        # setup file ops
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Save a copy of the data in the result folder so it is handy
        if hasattr(self.model,'data'):
            # np.savetxt(self.output_dir + '/' + self.name + '_data', self.model.data, newline="\n")
            np.savetxt(self.output_dir + '/' + self.name + '_data', self.model.data)

        # setup device
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda': torch.cuda.manual_seed(self.seed)

        print('')
        print('--- Running on device: '+ str(self.device))
        print('')

        # model
        if self.flow_type == 'maf':
            nf = MAF(self.n_blocks, self.input_size, self.hidden_size, self.n_hidden, None,
                     self.activation_fn, self.input_order, batch_norm=self.batch_norm_order)
        elif self.flow_type == 'realnvp':  # Under construction
            nf = RealNVP(self.n_blocks, self.input_size, self.hidden_size, self.n_hidden, None,
                         batch_norm=self.batch_norm_order)
        else:
            raise ValueError('Unrecognized model.')

        nf = nf.to(self.device)
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(nf.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(nf.parameters(), lr=self.lr)
        else:
            raise ValueError('Unrecognized optimizer.')

        if self.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_step, self.lr_decay)
        elif self.lr_scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)
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
                self.n_iter = self.T
                self.batch_size = self.N
                
                if t == self.t0:
                    self.n_iter = self.T_0
                if t == 1:
                    self.batch_size = self.N_1
                    self.n_iter = self.T_1

                while i < prev_i + self.n_iter:
                    self.train(nf, optimizer, i, loglist, sampling=True, t=t)
                    if t == 1:
                        scheduler.step()
                    i += 1
                prev_i = i

                if (self.scheduler == 'AdaAnn'):
                    z0 = nf.base_dist.sample([self.M])
                    zk, _ = nf(z0)
                    log_qk = self.model_logdensity(zk)
                    dt = self.tol / torch.sqrt(log_qk.var())
                    dt = dt.detach()# .numpy()

                if (self.scheduler == 'Linear'):
                    dt = self.linear_step
        else:
            loglist = []
            for i in range(1, self.n_iter+1):                
                self.train(nf, optimizer, i, loglist, sampling=True)
                scheduler.step()

        print('')
        print('--- Simulation completed!')        

    def train(self, nf, optimizer, iteration, log, sampling=True, t=1):
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
        x0 = nf.base_dist.sample([self.batch_size])
        xk, sum_log_abs_det_jacobians = nf(x0)

        # Check for last iteration
        last_it = (iteration == self.n_iter)

        # generate and save samples evaluation
        if(sampling):
            if (iteration % self.save_interval == 0) or last_it:
                print('--- Saving results at iteration '+str(iteration))
                x00 = nf.base_dist.sample([self.n_sample])
                xkk, _ = nf(x00)
                
                # Save surrogate grid - there is no grid for discrepancy
                if self.surrogate and (self.surrogate_type == 'surrogate'):
                    np.savetxt(self.output_dir + '/' + self.name + '_grid_' + str(iteration), self.surrogate.grid_record.clone().cpu().numpy(), newline="\n")
                
                # Save log profile
                np.savetxt(self.output_dir + '/' + self.log_file, np.array(log), newline="\n")
                
                # Save normalized domain samples
                np.savetxt(self.output_dir + '/' + self.name + '_samples_' + str(iteration), xkk.data.clone().cpu().numpy(), newline="\n")
                
                # Save samples in the original space
                if self.transform:
                    xkk_samples = self.transform.forward(xkk).data.cpu().numpy()
                    np.savetxt(self.output_dir + '/' + self.name + '_params_' + str(iteration), xkk_samples, newline="\n")
                else:
                    xkk_samples = xkk.data.cpu().numpy()
                    np.savetxt(self.output_dir + '/' + self.name + '_params_' + str(iteration), xkk_samples, newline="\n")
                
                # Save marginal statistics
                np.savetxt(self.output_dir + '/' + self.name + '_marginal_stats_' + str(iteration), np.concatenate((xkk_samples.mean(axis=0).reshape(-1,1),xkk_samples.std(axis=0).reshape(-1,1)),axis=1), newline="\n")
                
                # Save log density at the same samples
                np.savetxt(self.output_dir + '/' + self.name + '_logdensity_' + str(iteration), self.model_logdensity(xkk).data.cpu().numpy(), newline="\n")
                
                # Save model outputs at the samples - If a model is defined
                if self.transform:
                    if(self.surrogate_type == 'surrogate'):
                        # Define noise when we use NoFAS
                        stds = torch.abs(self.model.defOut).to(self.device) * self.model.stdRatio
                        o00 = torch.randn(x00.size(0), self.model.data.shape[0]).to(self.device)
                        noise = o00*stds.repeat(o00.size(0),1)
                        # Compute outputs
                        if self.surrogate:
                            np.savetxt(self.output_dir + '/' + self.name + '_outputs_' + str(iteration), (self.surrogate.forward(xkk) + noise).data.cpu().numpy(), newline="\n")
                        else:
                            np.savetxt(self.output_dir + '/' + self.name + '_outputs_' + str(iteration), (self.model.solve_t(self.transform.forward(xkk)) + noise).data.cpu().numpy(), newline="\n")
                    elif(self.surrogate_type == 'discrepancy'):
                        # Define noise when we use NoFAS
                        stds = torch.abs(self.model.defOut).to(self.device) * self.model.stdRatio
                        # Noise is rows: number of T,P pairs, columns: number of batches
                        o00 = torch.randn(self.model.data.shape[0], x00.size(0)).to(self.device)
                        noise = o00*stds.repeat(1,x00.size(0))
                        # Print lf outputs
                        model_out = self.model.solve_t(self.transform.forward(xkk))
                        np.savetxt(self.output_dir + '/' + self.name + '_outputs_lf_' + str(iteration), model_out.data.cpu().numpy(), newline="\n")                    
                        # LF model, plus dicrepancy, plus noise
                        if(self.surrogate is None):
                            # This need to have as many rows as T,P
                            # and as many columns as batches                        
                            model_out_noise = model_out + noise
                            np.savetxt(self.output_dir + '/' + self.name + '_outputs_lf+noise_' + str(iteration), model_out_noise.data.cpu().numpy(), newline="\n")
                        else:
                            discr_out = self.surrogate.forward(self.model.var_in)
                            # CHECK COMPATIBILITY !!!
                            model_out_lf_discr = model_out + discr_out                        
                            model_out_lf_discr_noise = model_out + discr_out + noise
                            # Save model outputs
                            # For discrepancy we have
                            # Rows: number of variable pairs
                            # Columns: number of batches
                            np.savetxt(self.output_dir + '/' + self.name + '_outputs_discr_' + str(iteration), discr_out.data.cpu().numpy(), newline="\n")
                            np.savetxt(self.output_dir + '/' + self.name + '_outputs_lf+discr_' + str(iteration), model_out_lf_discr.data.cpu().numpy(), newline="\n")
                            np.savetxt(self.output_dir + '/' + self.name + '_outputs_lf+discr+noise_' + str(iteration), model_out_lf_discr_noise.data.cpu().numpy(), newline="\n")
                    else:
                        print('Invalid type of surrogate model')
                        exit(-1)


        if torch.any(torch.isnan(xk)):
            print("Error: samples xk are nan at iteration " + str(iteration))
            print(xk)
            np.savetxt(self.output_dir + '/' + self.log_file, np.array(log), newline="\n")
            exit(-1)

        # updating surrogate model
        if(self.run_nofas):
            if (iteration % self.calibrate_interval == 0) or last_it:
                if(self.surrogate_type == 'surrogate'):
                    go_on = self.surrogate.grid_record.size(0) < self.budget
                elif(self.surrogate_type == 'discrepancy'):
                    go_on = True
                else:
                    print('Invalid type of surrogate model')
                    exit(-1)
                if(go_on):    
                    # Update Surrogate Model
                    if(self.surrogate_type == 'surrogate'):
                        xk0 = xk[:self.true_data_num, :].data.clone()
                        self.surrogate.update(xk0, max_iters=self.surr_upd_it)
                    elif(self.surrogate_type == 'discrepancy'):
                        xk0 = xk.data.clone()
                        self.surrogate.update(self.transform.forward(xk0), max_iters=self.surr_upd_it, reg=False, reg_penalty=0.0001)
                    else:
                        print('Invalid type of surrogate model')
                        exit(-1)    

        # Free energy bound
        if(self.model_logprior is None):
            loss = (- torch.sum(sum_log_abs_det_jacobians, 1) - t * self.model_logdensity(xk)).mean()
        else:
            loss = (- torch.sum(sum_log_abs_det_jacobians, 1) - t * (self.model_logdensity(xk) + self.model_logprior(xk))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % self.log_interval == 0:
            print('VI NF (t=%5.3f): it: %7d | loss: %8.3e' % (t,iteration, loss.item()))
            log.append([t, iteration, loss.item()])
        
        # Save state of normalizing flow layers
        if self.store_nf_interval > 0 and iteration % self.store_nf_interval == 0:
            torch.save(nf.state_dict(), self.output_dir + '/' + self.name + "_" + str(iteration) + ".nf")

        if not(self.store_surr_interval is None) and (self.store_surr_interval > 0) and ((iteration % self.store_surr_interval == 0) or (last_it)):
            self.surrogate.surrogate_save() # Save surrogate model