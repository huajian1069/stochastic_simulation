import numpy as np
import scipy.stats as st
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.integrate as integrate
import dill as pickle
import os.path
import warnings
from statsmodels.tsa.stattools import acf

warnings.simplefilter("ignore")


class ParallelTempering:
    def __init__(self, D, K, N, p, u, u0, Ns, alpha_opt=None):
        """
        parallel tempering algorithm
        :param D: dimension of random vectors
        :param K: number of parallel temperature
        :param N: length of returned random number array
        :param p: the transition density
        :param u: target density in different temperature
        :param u0: initial distribution
        :param Ns: every Ns steps, conduct one step of swapping
        :param alpha_opt: desired acceptance rate
        :return: the generated Markov chain, acceptance rate of swapping states
        """
        self.D = D
        self.K = K
        self.N = N
        self.p = p
        self.u = u
        self.u0 = u0
        self.Ns = Ns
        self.alpha_opt = alpha_opt
        print()

    def generateMarkovChain(self, mode):
        acc = None
        self.mode = mode
        if mode == 'simple PT':
            self.xs, acc = simple_parallel_tempering(self.D, self.K, self.N, self.p, self.u, self.u0, self.Ns)
        elif mode == 'full PT':
            self.xs, acc = full_parallel_tempering(self.D, self.K, self.N, self.p, self.u, self.u0, self.Ns)
        elif mode == 'adaptive PT':
            self.xs, acc = adaptive_parallel_tempering(self.D, self.K, self.N, self.p, self.u, self.u0, self.Ns,
                                                       self.alpha_opt)
        elif mode == 'without PT':
            self.xs = random_walk_metropolis(self.D, self.N, self.p[0], self.u[0], self.u0)
        else:
            raise ValueError("Unsupported mode, please check your spelling and "
                             "select one from {'simple PT', 'full PT', 'without PT'} ")
        return self.xs, acc

    def load(self, address):
        if os.path.exists(address):
            with open(address, 'rb') as f:
                self.__dict__.update(pickle.Unpickler(f).load().__dict__)
        else:
            raise OSError('data is not found on disk, please generate first')

    def save(self, address):
        with open(address, "wb") as f:
            pickle.dump(self, f)

    def plot_hist(self, ax, density_des):
        u = self.u[0]
        x_t = np.linspace(-5, 5, 1000)
        y_t = u(x_t)
        integral = integrate.quad(u, -5, 5)[0]

        ax.hist(self.xs, bins=100, density=True, label=self.mode)
        ax.plot(x_t, y_t / integral, linewidth=1.5, label=r'$\mu_1(\theta)$')
        ax.set_title(density_des)
        plt.legend()

    def plot_hist2d(self, ax, density_des):
        xmin, xmax = -2, 13
        ymin, ymax = -1, 3
        ax.hist2d(self.xs[:, 0], self.xs[:, 1], bins=[150, 40], norm=mcolors.PowerNorm(0.3),
                  range=[[xmin, xmax], [ymin, ymax]])
        ax.set_title(density_des + '+' + self.mode)

    def plot_acf(self, ax, density_des):
        r_xs = acf(self.xs)
        ax.bar(range(len(r_xs)), r_xs, label=self.mode)
        ax.set_title(density_des)
        plt.legend()

    def plot_acf2d(self, axs, density_des, k=41):
        r_theta1 = acf(self.xs[:, 0], k)
        r_theta2 = acf(self.xs[:, 1], k)
        axs[0].bar(range(len(r_theta1)), r_theta1, label='theta1')
        axs[0].set_title(density_des + '+' + self.mode)
        axs[0].legend()
        axs[1].bar(range(len(r_theta2)), r_theta2, label='theta2')
        axs[1].set_title(density_des + '+' + self.mode)
        axs[1].legend()

    def plot_trace(self, ax, density_des):
        ax.plot(self.xs, label=density_des)
        ax.set_title(self.mode)
        plt.legend()

    def plot_trace2d(self, ax, density_des):
        ax.plot(self.xs[:, 0], label='theta1')
        ax.plot(self.xs[:, 1], label='theta2')
        ax.set_title(density_des + '+' + self.mode)
        ax.legend()

    def get_effective_sample_size(self):
        return np.abs(self.N / (1+2*(np.abs(acf(self.xs, 1000)).sum()-1)))

    def get_effective_sample_size_2d(self):
        return np.abs(self.N / (1+2*(np.abs(acf(self.xs[:, 0], 1000)).sum()-1))), np.abs(
            self.N / (1+2*(np.abs(acf(self.xs[:, 1], 1000)).sum()-1)))


def metropolis_hastings(x, p, u):
    """
    generate proposal from p, accept or reject it based on calculating the
    :param x: current state
    :param p: the transition density
    :param u: target density
    :return: next state, at the end, this function return a equivalent desired invariant distribution
    """
    y = p(loc=x)
    if st.uniform.rvs() < u(y) / u(x):
        x = y
    return x


def random_walk_metropolis(D, N, p, u, u0):
    """
    random walk Metropolis algorithm
    :param D: dimension of random vectors
    :param N: length of returned random number array
    :param p: the transition density
    :param u: target density in different temperature
    :param u0: initial distribution
    :return: the generated Markov chain
    """
    x = np.squeeze(np.zeros((N, D)))
    x[0] = u0()
    for n in range(N - 1):
        x[n + 1] = metropolis_hastings(x[n], p, u)
    return x


def simple_parallel_tempering(D, K, N, p, u, u0, Ns):
    """
    parallel tempering algorithm
    :param D: dimension of random vectors
    :param K: number of parallel temperature
    :param N: length of returned random number array
    :param p: the transition density
    :param u: target density in different temperature
    :param u0: initial distribution
    :param Ns: every Ns steps, conduct one step of swapping
    :return: the generated Markov chain, acceptance rate of swapping states
    """
    acceptance = 0
    x = np.squeeze(np.zeros((K, N, D)))
    for i in range(K):
        x[i, 0] = u0()
    for n in range(N - 1):
        for k in range(K):
            x[k, n + 1] = metropolis_hastings(x[k, n], p[k], u[k])
        if n % Ns == 0:
            i = int(st.uniform.rvs() * (K - 1))
            ratio = u[i + 1](x[i, n + 1]) * u[i](x[i + 1, n + 1]) / (u[i](x[i, n + 1]) * u[i + 1](x[i + 1, n + 1]))
            if st.uniform.rvs() < ratio:
                x[i, n + 1], x[i + 1, n + 1] = x[i + 1, n + 1], x[i, n + 1]
                acceptance += 1
    return x[0], acceptance / ((N - 1) / Ns)


def full_parallel_tempering(D, K, N, p, u, u0, Ns):
    """
    parallel tempering algorithm
    :param D: dimension of random vectors
    :param K: number of parallel temperature
    :param N: length of returned random number array
    :param p: the transition density
    :param u: target density in different temperature
    :param u0: initial distribution
    :param Ns: every Ns steps, conduct one step of swapping
    :return: desired random number sampled from desired distribution
    """
    acceptance = 0
    x = np.squeeze(np.zeros((K, N, D)))
    for i in range(K):
        x[i, 0] = u0()
    for n in range(N - 1):
        for k in range(K):
            x[k, n + 1] = metropolis_hastings(x[k, n], p[k], u[k])
        if n % Ns == 0:
            for i in range(K - 1):
                ratio = u[i + 1](x[i, n + 1]) * u[i](x[i + 1, n + 1]) / (u[i](x[i, n + 1]) * u[i + 1](x[i + 1, n + 1]))
                if st.uniform.rvs() < ratio:
                    x[i, n + 1], x[i + 1, n + 1] = x[i + 1, n + 1], x[i, n + 1]
                    acceptance += 1
    return x[0], acceptance / ((K - 1) * (N - 1) / Ns)


def adaptive_parallel_tempering(D, K, N, p, u, u0, Ns, alpha_opt):
    """
    parallel tempering algorithm
    :param D: dimension of random vectors
    :param K: number of parallel temperature
    :param N: length of returned random number array
    :param p: proposal distribution
    :param u: target density in different temperature
    :param u0: initial distribution
    :param Ns: every Ns steps, conduct one step of swapping
    :param alpha_opt: desired acceptance rate
    :return: the generated Markov chain, acceptance rate of swapping states
    """
    acceptance = 0
    x = np.squeeze(np.zeros((K, N, D)))
    logT = np.zeros((K - 1,))
    T = np.zeros((K,))
    T[0] = 1
    for i in range(K):
        x[i, 0] = u0()
    for i in range(K - 1):
        logT[i] = 1
        T[i + 1] = T[i] + np.exp(logT[i])
    for n in range(N - 1):
        for k in range(K):
            ut = lambda x: u[k](x, T[k])
            x[k][n + 1] = metropolis_hastings(x[k][n], p[k], ut)
        if n % Ns == 0:
            i = st.randint(0, K - 1).rvs()
            ratio = u[i + 1](x[i, n + 1], T[i + 1]) * u[i](x[i + 1, n + 1], T[i]) \
                    / (u[i](x[i, n + 1], T[i]) * u[i + 1](x[i + 1, n + 1], T[i + 1]))
            alpha = min(1, ratio)
            if st.uniform.rvs() < alpha:
                x[i, n + 1], x[i + 1, n + 1] = x[i + 1, n + 1], x[i, n + 1]
                acceptance += 1
            logT[i] += 0.6 / (n + 1) * (alpha - alpha_opt)
            T[i + 1] = T[i] + np.exp(logT[i])
    return x[0], acceptance / ((N - 1) / Ns)

a = np.array([-2, -1])
b = np.array([13, 3])
# np.random.seed(np.random.randint(1)+comm.rank)
signoise = np.array([0.05])
np.random.seed(12)


def L(x):
    """
    Defines likelihood functions
    :param x: 2D random vector
    :return: log likelihood
    """
    m = (1.01 / (3.142 - 4.9))
    C = -0.5 * (((x[0] - 1.7) ** 2 + (x[1] - 1) ** 2 - 1.0) ** 2) / (signoise ** 2) + np.log((x[0] <= 2.5))
    I1 = -0.5 * ((x[1] - 0) ** 2) / (signoise ** 2) + np.log(9 <= x[0] < 11)
    I2 = -0.5 * ((x[1] - 2) ** 2) / (signoise ** 2) + np.log(9 <= x[0] < 11)
    I3 = -0.5 * ((x[0] - 10) ** 2) / (signoise ** 2) + np.log(0 <= x[1] < 2)
    Q1 = -0.5 * (((x[0] - 7) ** 2 + (x[1] - 1) ** 2 - 1.0) ** 2) / (signoise ** 2)
    Q2 = -0.5 * ((x[1] + x[0] - 8) ** 2) / (signoise ** 2) + np.log(7.7 < x[0] < 8.4)
    S1 = -0.5 * ((x[1] - (m * x[0] + (1.5 - m * 3.142))) ** 2) / (signoise ** 2) + np.log(3.142 < x[0] < 4.86)
    S2 = -0.5 * (((x[0] - 4) ** 2 + (x[1] - 1) ** 2 - 1) ** 2) / (signoise ** 2) + np.log(x[1] < 0.5)
    S3 = -0.5 * (((x[0] - 4) ** 2 + (x[1] - 1) ** 2 - 1) ** 2) / (signoise ** 2) + np.log(x[1] > 1.5)
    return logsumexp([C, I1, I2, I3, Q1, Q2, S1, S2, S3])


def pr(x):
    """
    Defines prior functions
    :param x: 2D random vector
    :return: log prior probability
    """
    return np.log((x[0] >= a[0]) * (x[0] <= b[0]) * (x[1] >= a[1]) * (x[1] <= b[1]))


def post(x):
    """
    Define posterior function
    :param x: 2D random vector
    :return: log posterior probability
    """
    return L(x) + pr(x)
