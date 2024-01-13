# # True process posterior
# python3 -m linfa.plot_disc --folder results/ \
# 							--name test_18_lf_w_disc_TP15_prior_rep_meas \
# 							--iter 50000 \
# 							--mode histograms \
# 							--num_points 10 \
# 							--limfactor 1.0 \

# # Discrepancy surface
# python3 -m linfa.plot_disc --folder results/ \
# 							--name test_18_lf_w_disc_TP15_prior_rep_meas \
# 							--iter 50000 \
# 							--mode discr_surface \
# 							--num_points 40 \
# 							--limfactor 1.0 \

# # Marginal statistics
# python3 -m linfa.plot_disc --folder results/ \
# 							--name test_18_lf_w_disc_TP15_prior_rep_meas \
# 							--iter 50000 \
# 							--mode marginal_stats \
# 							--saveinterval 1000 \

# # Marginal posterior
# python3 -m linfa.plot_disc --folder results/ \
# 							--name test_18_lf_w_disc_TP15_prior_rep_meas \
# 							--iter 50000 \
# 							--mode marginal_posterior \

# # Loss vs. iterations plot, calibration parameters posterior
# python3 -m linfa.plot_res --folder results/ \
# 						--name test_18_lf_w_disc_TP15_prior_rep_meas \
# 						--iter 50000 \

Marginal statistics for groups to study
python3 -m linfa.plot_disc_groups --folder results/ \
								--names 'test_07_lf_w_disc_TP15' 'test_09_lf_w_disc_TP15_prior' 'test_16_lf_w_disc_TP15_rep_meas' 'test_18_lf_w_disc_TP15_prior_rep_meas' \
								--iter 50000 \
								--mode marginal_stats_group \
								--saveinterval 1000 \
								--labels 'TP15' 'TP15 + prior' 'TP15 + rep. meas.' 'TP15 + prior + rep. meas.' \
								--grouping 'prior'