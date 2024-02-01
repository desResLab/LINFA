python3 linfa/plot_res.py --folder results/ --name test_07_lf_w_disc_TP15 --iter 25000 --picformat png

python3 linfa/plot_disc.py --folder results/ --name test_07_lf_w_disc_TP15 --iter 25000 --picformat png --mode histograms --num_points 10 --limfactor 1.0 --saveinterval 1000
python3 linfa/plot_disc.py --folder results/ --name test_07_lf_w_disc_TP15 --iter 25000 --picformat png --mode discr_surface --num_points 10 --limfactor 1.0 --saveinterval 1000
python3 linfa/plot_disc.py --folder results/ --name test_07_lf_w_disc_TP15 --iter 25000 --picformat png --mode marginal_stats --num_points 10 --limfactor 1.0 --saveinterval 1000
python3 linfa/plot_disc.py --folder results/ --name test_07_lf_w_disc_TP15 --iter 25000 --picformat png --mode marginal_posterior --num_points 10 --limfactor 1.0 --saveinterval 1000





