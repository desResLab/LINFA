import numpy as np
import matplotlib.pyplot as plt

folder = 'results/test_out_with_disc/'
step_num = 501
plot_all_LF = False


obs_file = 'test_out_with_disc_data'
lf_file = 'test_out_with_disc_outputs_lf_' + str(step_num)
lf_discr_file = 'test_out_with_disc_outputs_lf+discr_' + str(step_num)
discr_file = 'test_out_with_disc_outputs_discr_' + str(step_num)

obs = np.loadtxt(folder + obs_file)
lf = np.loadtxt(folder + lf_file)
lf_discr = np.loadtxt(folder + lf_discr_file)
discr = np.loadtxt(folder + discr_file)

# Separate T,P from coverage
tp = obs[:,:2]
obs = obs[:,2:]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# True coverage
ax.scatter(tp[:,0],tp[:,1],obs,label='obs',color='b')

if(plot_all_LF):
	# LF
	for loopA in range(len(tp)):
		if(loopA == 0):
			ax.scatter(tp[:,0],tp[:,1],lf[:,loopA],color='r',label='LF',alpha=0.6)
		else:
			ax.scatter(tp[:,0],tp[:,1],lf[:,loopA],color='r',alpha=0.6)
	# DISCR
	for loopA in range(len(tp)):
		if(loopA == 0):
			ax.scatter(tp[:,0],tp[:,1],lf_discr[:,loopA],color='m',label='LF+DISC',alpha=0.6)
		else:
			ax.scatter(tp[:,0],tp[:,1],lf_discr[:,loopA],color='m',alpha=0.6)
else:
	# Plot average LF and Discrepancy
	ax.scatter(tp[:,0],tp[:,1],np.mean(lf,axis=1),color='r',label='LF',alpha=0.6)		
	ax.scatter(tp[:,0],tp[:,1],np.mean(lf_discr,axis=1),marker='D',color='m',label='LF+DISC',alpha=0.6)

plt.legend()

plt.show()

