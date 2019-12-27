import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


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


def metropolis_hastings_adaptive(x, p, u, T):
    """
    generate proposal from p, accept or reject it based on calculating the
    :param x: current state
    :param p: the transition density
    :param u: target density
    :param T: temperature of this parallel distribution
    :return: next state, at the end, this function return a equivalent desired invariant distribution
    """
    global acc
    y = p(loc=x)
    ratio = u(y, T) / u(x, T)
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
            ratio = u[i + 1](x[i][n + 1]) * u[i](x[i + 1][n + 1]) / (u[i](x[i][n + 1]) * u[i + 1](x[i + 1][n]))
            alpha = min(1, ratio)
            if st.uniform.rvs() < alpha:
                x_swap = x[i][n + 1]
                x[i][n + 1] = x[i + 1][n]
                x[i + 1][n] = x_swap
    return x[0]


def full_parallel_tempering(x, K, N, p, u, u0, Ns):
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
            for i in range(K-1):
                ratio = u[i + 1](x[i][n + 1]) * u[i](x[i + 1][n + 1]) / (u[i](x[i][n + 1]) * u[i + 1](x[i + 1][n]))
                alpha = min(1, ratio)
                if st.uniform.rvs() < alpha:
                    x_swap = x[i][n + 1]
                    x[i][n + 1] = x[i + 1][n]
                    x[i + 1][n] = x_swap
    return x[0]


def adaptive_parallel_tempering(x, K, N, p, u, u0, Ns, alpha_opt):
    """
    parallel tempering algorithm
    :param x: empty array of result random number
    :param K: number of parallel temperature
    :param N: length of returned random number array
    :param p: the transition density
    :param u: target density in different temperature
    :param u0: initial distribution
    :param Ns: every Ns steps, conduct one step of swapping
    :param alpha_opt: desired acceptance rate
    :return: desired random number sampled from desired distribution
    """
    logT = np.zeros((K-1, ))
    T = np.zeros((K, ))
    T[0] = 1
    for i in range(K):
        x[i][0] = u0()
    for i in range(K-1):
        logT[i] = 1
        T[i+1] = T[i] + np.exp(logT[i])
    for n in range(N - 1):
        for k in range(K):
            x[k][n + 1] = metropolis_hastings_adaptive(x[k][n], p[k], u[k], T[k])
        if n % Ns == 0:
            i = int(st.uniform.rvs() * (K - 1))
            ratio = u[i + 1](x[i][n + 1], T[i+1]) * u[i](x[i + 1][n + 1], T[i]) \
                    / (u[i](x[i][n + 1], T[i]) * u[i + 1](x[i + 1][n], T[i+1]))
            alpha = min(1, ratio)
            if st.uniform.rvs() < alpha:
                x_swap = x[i][n + 1]
                x[i][n + 1] = x[i + 1][n]
                x[i + 1][n] = x_swap
            logT[i] += 0.6/(n+1) * (alpha - alpha_opt)
        for i in range(K-1):
            T[i+1] = T[i] + np.exp(logT[i])
    return x[0]


acc = 0
K = 4
N = 10000
alpha_opt = 0.234
p = [lambda loc: st.norm.rvs(loc=loc, scale=3)] * K
gammas = [1, 2, 4, 8, 16]
a = 2
u0 = st.uniform(loc=-3, scale=6).rvs
Ns = 1

x = np.zeros((K, N))
x_t = np.linspace(-5, 5, 1000)

fig = plt.figure(figsize=(20, 4))
for j, gamma in enumerate(gammas):
    u = [lambda x, T: np.exp(-gamma * (x ** 2 - 1) ** 2 / T)] * K
    acc = 0
    xs = adaptive_parallel_tempering(x, K, N, p, u, u0, Ns, alpha_opt)
    stat = {'acceptance rate': acc / ((N - 1) * K)}
    print('acceptance rate when gamma=%d: %f' % (gamma, stat['acceptance rate']))
    y_t = u[0](x_t, 1)

    ax = fig.add_subplot(2, 5, j + 1)
    ax.hist(xs, bins=100, density=True, label='adaptive_PT')
    ax.plot(x_t, 0.22 * y_t / stat['acceptance rate'], linewidth=1., label=r'$\tilde{f}(x)$')
    ax.set_title('gamma = ' + str(gamma))

    acc = 0
    u = [lambda x, i=i: np.exp(-gamma * (x ** 2 - 1) ** 2 / (a ** i)) for i in range(K)]
    xs = full_parallel_tempering(x, K, N, p, u, u0, Ns)
    stat = {'acceptance rate': acc / ((N - 1) * K)}
    print('acceptance rate when gamma=%d: %f' % (gamma, stat['acceptance rate']))
    ax2 = fig.add_subplot(2, 5, 5 + j + 1)
    ax2.hist(xs, bins=100, density=True, label='simple_PT')
    ax2.plot(x_t, 0.22 * y_t / stat['acceptance rate'], linewidth=1., label=r'$\tilde{f}(x)$')
    ax2.set_title('gamma = ' + str(gamma))
plt.legend()
plt.savefig('figures/project_hist_ex4_adaptive.png')
plt.show()


fig = plt.figure(figsize=(20, 4))
for j, gamma in enumerate(gammas):
    u = [lambda x, i=i: np.exp(-gamma * (x ** 2 - 1) ** 2 / (a ** i)) for i in range(K)]
    acc = 0
    xs = full_parallel_tempering(x, K, N, p, u, u0, Ns)
    stat = {'acceptance rate': acc / ((N - 1) * K)}
    print('acceptance rate when gamma=%d: %f' % (gamma, stat['acceptance rate']))
    y_t = u[0](x_t)

    ax = fig.add_subplot(2, 5, j + 1)
    ax.hist(xs, bins=100, density=True, label='full_PT')
    ax.plot(x_t, 0.22 * y_t / stat['acceptance rate'], linewidth=1., label=r'$\tilde{f}(x)$')
    ax.set_title('gamma = ' + str(gamma))

    acc = 0
    xs = full_parallel_tempering(x, K, N, p, u, u0, Ns)
    stat = {'acceptance rate': acc / ((N - 1) * K)}
    print('acceptance rate when gamma=%d: %f' % (gamma, stat['acceptance rate']))
    ax2 = fig.add_subplot(2, 5, 5 + j + 1)
    ax2.hist(xs, bins=100, density=True, label='simple_PT')
    ax2.plot(x_t, 0.22 * y_t / stat['acceptance rate'], linewidth=1., label=r'$\tilde{f}(x)$')
    ax2.set_title('gamma = ' + str(gamma))

plt.legend()
# plt.savefig('figures/project_hist_ex4_full.png')
plt.show()
