# discr_surface
python3 -m linfa.plot_disc --folder results/ \
							--name test_07_lf_w_disc_TP15 \
							--iter 25000 \
							--mode histograms \
							--num_points 10 \
							--limfactor 1.0 \

python3 -m linfa.plot_disc --folder results/ \
							--name test_07_lf_w_disc_TP15 \
							--iter 25000 \
							--mode discr_surface \
							--num_points 40 \
							--limfactor 1.0 \

python3 -m linfa.plot_disc --folder results/ \
							--name test_07_lf_w_disc_TP15 \
							--iter 25000 \
							--mode marginal_stats \
							--saveinterval 1000 \

# python3 -m linfa.plot_disc_groups --folder results/ \
# 								--names 'test_15_lf_w_disc_TP1_rep_meas' 'test_16_lf_w_disc_TP15_rep_meas' 'test_17_lf_w_disc_TP1_prior_rep_meas' 'test_18_lf_w_disc_TP15_prior_rep_meas' \
# 								--iter 25000 \
# 								--mode marginal_stats_group \
# 								--saveinterval 1000 \
# 								--labels 'TP1' 'TP15' 'TP1 + prior' 'TP15 + prior' \
# 								--grouping 'repeated measurements'

python3 -m linfa.plot_res --folder results/ \
							--name test_07_lf_w_disc_TP15 \
							--iter 25000 \