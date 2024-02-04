import os,torch
from linfa.discrepancy import Discrepancy
from linfa.maf import MAF, RealNVP
from run_experiment import load_exp_from_file

def eval_discrepancy(file_path,test_data):

    # Read in data
    exp_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)

    # Create new discrepancy
    dicr = Discrepancy(model_name = exp_name, 
                       model_folder = dir_name,
                       lf_model = None,
                       input_size = None,
                       output_size = None,
                       var_grid_in = None,
                       var_grid_out = None)
    dicr.surrogate_load()

    # Evaluate discrepancy over test grid
    return dicr.forward(test_data) 

def eval_model(exp_chkpt_file,nf_chkpt_file,discr_chkpt_file,num_calib_samples,test_data):

    # Load experiment from file
    exp = load_exp_from_file(exp_chkpt_file)

    # Create NF model from experiment
    if exp.flow_type == 'maf':
        nf = MAF(exp.n_blocks, exp.input_size, exp.hidden_size, exp.n_hidden, None,
                    exp.activation_fn, exp.input_order, batch_norm=exp.batch_norm_order)
    elif exp.flow_type == 'realnvp':  # Under construction
        nf = RealNVP(exp.n_blocks, exp.input_size, exp.hidden_size, exp.n_hidden, None,
                        batch_norm=exp.batch_norm_order)

    # Read state dictionary    
    nf.state_dict(torch.load(nf_chkpt_file))        
    
    # Sample calibration parameter realizations
    x00 = nf.base_dist.sample([num_calib_samples])
    xkk, _ = nf(x00)

    # Evaluate discrepancy at tp data "test_data"
    res_discr = eval_discrepancy(discr_chkpt_file,test_data)

    # Solve models 
    # Need to change this to be evaluated at arbitraty temperatures and pressures.
    if(exp.transform is None):
        res_lf = exp.model.solve_t(xkk)
    else:
        res_lf = exp.model.solve_t(exp.transform.forward(xkk))

    # CURRENTLY NO NOISE IS ADDED, NEED TO BE IMPLEMENTED IF APPROPRIATE!!!

    # return 
    return res_lf + res_discr

# MAIN CODE
if __name__ == "__main__":

    # Assign files
    exp_chkpt_file = './tests/results/test_lf_with_disc_hf_data_TP1/experiment.pt'
    nf_chkpt_file = './tests/results/test_lf_with_disc_hf_data_TP1/test_lf_with_disc_hf_data_TP1_3000.nf'
    discr_chkpt_file =  './tests/results/test_lf_with_disc_hf_data_TP1/test_lf_with_disc_hf_data_TP1'
    # 
    num_calib_samples = 100

    # Set the grid for 
    min_dim_1 = 400.0
    max_dim_1 = 500.0
    min_dim_2 = 2.0
    max_dim_2 = 3.0
    num_1d_grid_points = 5
    #
    test_grid_1 = torch.linspace(min_dim_1, max_dim_1, num_1d_grid_points)
    test_grid_2 = torch.linspace(min_dim_2, max_dim_2, num_1d_grid_points)
    grid_t, grid_p = torch.meshgrid(test_grid_1, test_grid_2, indexing='ij')
    test_data = torch.cat((grid_t.reshape(-1,1), grid_p.reshape(-1,1)),1)

    res = eval_model(exp_chkpt_file,nf_chkpt_file,discr_chkpt_file,num_calib_samples,test_data)

    print(res.size())