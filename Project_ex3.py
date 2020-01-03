import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import matplotlib.colors as mcolors
import warnings
warnings.simplefilter("ignore")


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
            ratio = u[i + 1](x[i][n + 1]) * u[i](x[i + 1][n + 1]) / (u[i](x[i][n + 1]) * u[i + 1](x[i + 1][n]))
            alpha = min(1, ratio)
            if st.uniform.rvs() < alpha:
                x_swap = x[i][n + 1]
                x[i][n + 1] = x[i + 1][n]
                x[i + 1][n] = x_swap
    return x[0]


a = np.array([-2, -1])
b = np.array([13, 3])
# np.random.seed(np.random.randint(1)+comm.rank)
signoise = np.array([0.05])


# Defines likelihood, prior and posterior functions
def L(x):
    # C=\left(x-1.7\right)^2+\left(y-1\right)^2=1\left\{x\le1.8+.7\right\}
    m = (1.01 / (3.142 - 4.9))
    C = -0.5 * (((x[0] - 1.7) ** 2 + (x[1] - 1) ** 2 - 1.0) ** 2) / (signoise ** 2) + np.log((x[0] <= 2.5))
    I1 = -0.5 * ((x[1] - 0) ** 2) / (signoise ** 2) + np.log(x[0] < 11) + np.log(x[0] >= 9)
    I2 = -0.5 * ((x[1] - 2) ** 2) / (signoise ** 2) + np.log(x[0] < 11) + np.log(x[0] >= 9) + np.log(
        x[1] >= 0) + np.log(x[1] < 2)
    I3 = -0.5 * ((x[0] - 10) ** 2) / (signoise ** 2) + np.log(x[1] < 2) + np.log(x[0] >= 0)
    Q1 = -0.5 * (((x[0] - 7) ** 2 + (x[1] - 1) ** 2 - 1.0) ** 2) / (signoise ** 2)
    Q2 = -0.5 * ((x[1] + x[0] - 8) ** 2) / (signoise ** 2) + np.log(x[0] < 8.4) + np.log(x[0] > 7.7)

    S1 = -0.5 * ((x[1] - (m * x[0] + (1.5 - m * 3.142))) ** 2) / (signoise ** 2) + np.log(x[0] < 4.86) + np.log(
        x[0] > 3.142)

    S2 = -0.5 * (((x[0] - 4) ** 2 + (x[1] - 1) ** 2 - 1) ** 2) / (signoise ** 2) + np.log(x[1] < 0.5)
    S3 = -0.5 * (((x[0] - 4) ** 2 + (x[1] - 1) ** 2 - 1) ** 2) / (signoise ** 2) + np.log(x[1] > 1.5)
    return logsumexp([C, I1, I2, I3, Q1, Q2, S1, S2, S3]);


def pr(x):
    return np.log((x[0] >= a[0]) * (x[0] <= b[0]) * (x[1] >= a[1]) * (x[1] <= b[1]));


def post(x):
    return L(x) + pr(x)


acc = 0
K = 4
Ta = 2
N = 10000
Ns = 1
u = [lambda x, i=i: np.exp(post(x) / Ta ** i) for i in range(4)]
p = [lambda loc: st.multivariate_normal.rvs(mean=loc, cov=np.array([[1, 0], [0, 0.5]]))] * K
u0 = st.uniform(loc=np.array([-2, -1]), scale=np.array([15, 4])).rvs
x = np.zeros((K, N, 2))

x0 = simple_parallel_tempering(x, K, N, p, u, u0, Ns)
stat = {'acceptance rate': acc / ((N - 1) * K)}
print('acceptance rate: %f' % (stat['acceptance rate']))
xmin = -2
xmax = 13
ymin = -1
ymax = 3
fig = plt.figure(figsize=(15, 4))
plt.hist2d(x0[:, 0], x0[:, 1], bins=[150, 40], norm=mcolors.PowerNorm(0.3), range=[[xmin, xmax], [ymin, ymax]])
plt.savefig('figures/csqi_hist2d.png')
plt.show()

