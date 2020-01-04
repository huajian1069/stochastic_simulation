import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def metropolis_hastings(x, p, u):
    """
    generate proposal from p, accept or reject it based on calculating the
    :param x: current state
    :param p: the transition density
    :param u: target density
    :return: next state, at the end, this function return a equivalent desired invariant distribution
    """
    global acc
    y = p(loc=x)
    ratio = u(y) / u(x)
    alpha = min(1, ratio)
    if st.uniform.rvs() < alpha:
        x_next = y
        acc += 1
    else:
        x_next = x
    return x_next


def simple_parallel_tempering(x, K, N, p, u, u0, Ns):
    """
    parallel tempering algorithm
    :param x: empty array of result random number
    :param K: number of parallel temperature
    :param N: length of returned random number array
    :param p: the transition density
    :param u: target density in different temperature
    :param u0: initial distribution
    :param Ns: every Ns steps, conduct one step of swapping
    :return: desired random number sampled from desired distribution
    """
    for i in range(K):
        x[i][0] = u0()
    for n in range(N - 1):
        for k in range(K):
            x[k][n + 1] = metropolis_hastings(x[k][n], p[k], u[k])
        if n % Ns == 0:
            i = int(st.uniform.rvs() * (K - 1))
            ratio = u[i + 1](x[i][n + 1]) * u[i](x[i + 1][n + 1]) / (u[i](x[i][n + 1]) * u[i + 1](x[i + 1][n + 1]))
            alpha = min(1, ratio)
            if st.uniform.rvs() < alpha:
                x_swap = x[i][n + 1]
                x[i][n + 1] = x[i + 1][n]
                x[i + 1][n] = x_swap
    return x[0]


def random_walk_metropolis(N, p, u, u0):
    """
    random walk Metropolis algorithm
    :param N: length of returned random number array
    :param p: the transition density
    :param u: target density in different temperature
    :param u0: initial distribution
    :return: desired random number sampled from desired distribution
    """
    x = np.zeros((N,))
    x[0] = u0()
    for n in range(N - 1):
        x[n + 1] = metropolis_hastings(x[n], p, u)
    return x

def acf(x, k = 41):
    """
    compute the auto-correlations
    :param x: the generated Markov chain
    :param k: lag size
    """
    N = len(x)
    r = np.zeros((k, ))
    m = x.mean()
    for i in range(k):
        r[i] = np.correlate(x[:N-i] - m, x[i:] - m) / np.sqrt(sum((x[:N-i] - m)**2) * sum((x[i:] - m)**2))
    return r

acc = 0
K = 4
N = 10000
p = [lambda loc: st.norm.rvs(loc=loc, scale=3)] * K
gammas = [1, 2, 4, 8, 16]
a = 2
u0 = st.uniform(loc=-3, scale=6).rvs
Ns = 1
x = np.zeros((K, N))
x_t = np.linspace(-5, 5, 1000)
# plot the histograms of samples
fig = plt.figure(figsize=(20, 7))
for j, gamma in enumerate(gammas):
    u = [lambda x, i=i: np.exp(-gamma * (x ** 2 - 1) ** 2 / (a ** i)) for i in range(K)]
    acc = 0
    xs = simple_parallel_tempering(x, K, N, p, u, u0, Ns)
    xs_walk = random_walk_metropolis(N, p[0], u[0], u0)
    stat = {'acceptance rate': acc / ((N - 1) * K)}
    print('acceptance rate when gamma=%d: %f' % (gamma, stat['acceptance rate']))
    y_t = u[0](x_t)
    integral = integrate.quad(u[0], -5, 5)[0]
    
    ax = fig.add_subplot(2, 5, j + 1)
   # ax.hist(xs_walk, bins=100, density=True, label='Walk')
    ax.hist(xs, bins=100, density=True, label='simple_PT')
    ax.plot(x_t, y_t / integral, linewidth=1.5, label=r'$\tilde{f}(x)$')
    ax.set_title('gamma = ' + str(gamma))
    plt.legend()

    ax2 = fig.add_subplot(2, 5, 5+ j + 1)
    ax2.hist(xs_walk, bins=100, density=True, label='Walk')
    ax2.plot(x_t, y_t / integral, linewidth=1.5, label=r'$\tilde{f}(x)$')
    ax2.set_title('gamma = ' + str(gamma))
    plt.legend()
plt.savefig('figures/project_ex2_hist.png')
plt.show()

# plot the auto-correlation plots
x = np.zeros((K, N))
fig = plt.figure(figsize=(20, 20))
for j, gamma in enumerate(gammas):
    u = [lambda x, i=i: np.exp(-gamma * (x ** 2 - 1) ** 2 / (a ** i)) for i in range(K)]
    acc = 0
    xs = simple_parallel_tempering(x, K, N, p, u, u0, Ns)
    xs_walk = random_walk_metropolis(N, p[0], u[0], u0)
    r_xs = acf(xs)
    r_walk = acf(xs_walk)
    
    ax = fig.add_subplot(5, 2, 2*j + 1)
    ax.bar(range(len(r_xs)), r_xs, label='simple_PT')
    ax.set_title('gamma = ' + str(gamma))
    plt.legend()

    ax2 = fig.add_subplot(5, 2, 2*j + 2)
    ax2.bar(range(len(r_walk)), r_walk, label='Walk')
    ax2.set_title('gamma = ' + str(gamma))
    plt.legend()
plt.savefig('figures/project_ex2_auto_correlation.png')
plt.show()

# plot the trace-plots
x = np.zeros((K, N))
fig = plt.figure(figsize=(20, 30))
for j, gamma in enumerate(gammas):
    u = [lambda x, i=i: np.exp(-gamma * (x ** 2 - 1) ** 2 / (a ** i)) for i in range(K)]
    acc = 0
    xs = simple_parallel_tempering(x, K, N, p, u, u0, Ns)
    xs_walk = random_walk_metropolis(N, p[0], u[0], u0)
    
    ax = fig.add_subplot(10, 1, 2*j + 1)
    ax.plot(xs, label='simple_PT')
    ax.set_title('gamma = ' + str(gamma))
    plt.legend()

    ax2 = fig.add_subplot(10, 1, 2*j + 2)
    ax2.plot(xs_walk, label='Walk')
    ax2.set_title('gamma = ' + str(gamma))
    plt.legend()
plt.savefig('figures/project_ex2_trace_plot.png')
plt.show()

# compute the efficient sample size
x = np.zeros((K, N))
for j, gamma in enumerate(gammas):
    u = [lambda x, i=i: np.exp(-gamma * (x ** 2 - 1) ** 2 / (a ** i)) for i in range(K)]
    ess_xs = 0
    ess_walk = 0
    for l in range(5):
        acc = 0
        xs = simple_parallel_tempering(x, K, N, p, u, u0, Ns)
        xs_walk = random_walk_metropolis(N, p[0], u[0], u0)
        ess_xs += np.abs(N/(1+2*(acf(xs, 1000).sum()-1)))
        ess_walk += np.abs(N/(1+2*(acf(xs_walk, 1000).sum()-1)))
    print('effective sample size for Markov chain generated by simple PT: %f' % (ess_xs/5)) 
    print('effective sample size for Markov chain generated by random walk MH: %f' % (ess_walk/5))

'''
gamma = 1
effective sample size for Markov chain generated by simple PT: 2135.576968
effective sample size for Markov chain generated by random walk MH: 2496.629166

gamma = 2
effective sample size for Markov chain generated by simple PT: 7569.012008
effective sample size for Markov chain generated by random walk MH: 3011.531131

gamma = 4
effective sample size for Markov chain generated by simple PT: 4402.382063
effective sample size for Markov chain generated by random walk MH: 1481.703287

gamma = 8
effective sample size for Markov chain generated by simple PT: 1601.226823
effective sample size for Markov chain generated by random walk MH: 862.148858

gamma = 16
effective sample size for Markov chain generated by simple PT: 8534.720071
effective sample size for Markov chain generated by random walk MH: 388.920264
'''