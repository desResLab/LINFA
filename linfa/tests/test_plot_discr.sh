# # discr_surface
# python3 -m linfa.plot_disc --folder results/ \
# 							--name test_08_lf_w_disc_TP1_prior \
# 							--iter 25000 \
# 							--mode histograms \
# 							--num_points 10 \
# 							--limfactor 1.0 \

python3 -m linfa.plot_disc --folder results/ \
							--name test_18_lf_w_disc_TP15_prior_rep_meas \
							--iter 25000 \
							--mode discr_surface \
							--num_points 40 \
							--limfactor 1.0 \

# python3 -m linfa.plot_res --folder results/ \
# 							--name test_08_lf_w_disc_TP1_prior \
# 							--iter 25000 \