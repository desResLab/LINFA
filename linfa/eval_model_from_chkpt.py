import os,torch
from linfa.discrepancy import Discrepancy
from linfa.maf import MAF, RealNVP

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


def eval_model(exp,discr_chkpt_file,nf_chkpt_file,test_data):

    # Evaluate discrepancy at tp data "test_data"
    res_discr = eval_discrepancy(discr_chkpt_file,test_data)
    # Sample from normalzing flow

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

    # Solve models 
    res_lf = exp.model.solve_t(exp.transform.forward(xkk))

    # return 
    return res_lf + res_discr




    

    res_model = eval_discrepancy(file_path,test_data_discr)




