from Project_utils import *
import warnings
warnings.simplefilter("ignore")
np.random.seed(12)


D = 2
K = 4
N = 10000
p = [lambda loc: st.multivariate_normal.rvs(mean=loc, cov=np.array([[1, 0], [0, 0.5]]))] * K
u0 = st.uniform(loc=np.array([-2, -1]), scale=np.array([15, 4])).rvs
Ns = 1
T_factor = 2
u = [lambda x, T=k: np.exp(post(x) / T_factor ** T) for k in range(K)]
experiment_control = ParallelTempering(D, K, N, p, u, u0, Ns)
experiment_contrast = ParallelTempering(D, K, N, p, u, u0, Ns)
file_simple_name = 'data/ex3/simple_csqi.obj'
file_walk_name = 'data/ex3/walk_csqi.obj'

val = input('Enter y to read data from disk, n to regenerate data\n')
if val == 'n':
    # generate data from Markov Chain
    _, acc = experiment_control.generateMarkovChain(mode='simple PT')
    print('Simple PT: swapping acceptance rate: %f' % acc)
    experiment_contrast.generateMarkovChain(mode='without PT')
    experiment_control.save(file_simple_name)
    experiment_contrast.save(file_walk_name)
else:
    experiment_control.load(file_simple_name)
    experiment_contrast.load(file_walk_name)

# plot the histograms of samples
fig = plt.figure(figsize=(20, 7))
ax = fig.add_subplot(2, 1, 1)
experiment_control.plot_hist2d(ax, "multi-modal distribution")
ax = fig.add_subplot(2, 1, 2)
experiment_contrast.plot_hist2d(ax, "multi-modal distribution")
plt.savefig('figures/ex3/csqi_hist2d.png')
plt.show()

# plot the auto-correlation plots
fig = plt.figure(figsize=(20, 7))
ax1, ax3 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 3)
experiment_control.plot_acf2d([ax1, ax3], 'multi-modal distribution', k=100)
ax2, ax4= fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 4)
experiment_contrast.plot_acf2d([ax2, ax4], 'multi-modal distribution', k=100)
plt.savefig('figures/ex3/csqi_acf2d.png')
plt.show()

# plot the trace-plots
fig = plt.figure(figsize=(20, 7))
ax = fig.add_subplot(2, 1, 1)
experiment_control.plot_trace2d(ax, 'multi-modal distribution')
ax = fig.add_subplot(2, 1, 2)
experiment_contrast.plot_trace2d(ax, 'multi-modal distribution')
plt.savefig('figures/ex3/csqi_trace2d.png')
plt.show()

# compute the efficient sample size
max_iter = 5
ess_simple = experiment_control.get_effective_sample_size_2d()
ess_walk = experiment_contrast.get_effective_sample_size_2d()
for l in range(max_iter-1):
    experiment_control.generateMarkovChain(mode='simple PT')
    experiment_contrast.generateMarkovChain(mode='without PT')
    ess_simple += experiment_control.get_effective_sample_size_2d()
    ess_walk += experiment_contrast.get_effective_sample_size_2d()
print('average of ' + str(max_iter) + 'times of running')
print('effective sample size of (theta1, theta2) for Markov chain generated by simple PT: (%f, %f)' % (ess_simple[0] / max_iter, ess_simple[1] / max_iter))
print('effective sample size of (theta1, theta2) for Markov chain generated by random walk MH: (%f, %f)'% (ess_walk[0] / max_iter, ess_walk[1] / max_iter))
