import numpy as np
import scipy.stats as st
acc = 0


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


def random_walk_metropolis(x, N, p, u, u0):
    """
    random walk Metropolis algorithm
    :param x: empty array of result random number
    :param N: length of returned random number array
    :param p: the transition density
    :param u: target density in different temperature
    :param u0: initial distribution
    :return: desired random number sampled from desired distribution
    """
    x[0] = u0()
    for n in range(N - 1):
        x[n + 1] = metropolis_hastings(x[n], p, u)
    return x[0]


def acf(x, k=41):
    """
    compute the sample auto-correlations function (ACF)
    :param x: the generated Markov chain
    :param k: lag size
    """
    N = len(x)
    r = np.zeros((k,))
    x = (x - x.mean()) / x.std()
    for i in range(k):
        r[i] = np.correlate(x[:N - i], x[i:]) / (N - i)
    return r
