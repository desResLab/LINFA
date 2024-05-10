python3 linfa/plot_res.py --folder results/ --name TP1_no_disc_gaussian_prior_linear --iter 10000 --picformat png
# python3 linfa/plot_disc.py --folder results/ --name test_08_lf_w_disc_TP1_uniform_prior --iter 25000 --picformat png --mode histograms --num_points 10 --limfactor 1.0 --saveinterval 1000 --dropouts 10
# python3 linfa/plot_disc.py --folder results/ --name test_19_lf_w_disc_TP15_rep_meas_dropout --iter 10000 --picformat png --mode discr_surface --num_points 10 --limfactor 1.0 --saveinterval 1000
# python3 linfa/plot_disc.py --folder results/ --name test_08_lf_w_disc_TP1_uniform_prior --iter 25000 --picformat png --mode marginal_stats --num_points 10 --limfactor 1.0 --saveinterval 1000
python3 linfa/plot_disc.py --folder results/ --name TP1_no_disc_gaussian_prior_linear --iter 10000 --picformat png --mode marginal_posterior --num_points 10 --limfactor 1.0 --saveinterval 1000





