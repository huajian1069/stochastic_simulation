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


acc = 0
K = 4
a = 2
N = 100
Ns = 1
sigma = 0.05
phi1 = lambda x, y: ((x - 1.7) ** 2 + (y - 1) ** 2 - 1) ** 2 * (x <= 2.5) / (-2 * sigma ** 2)
phi2 = lambda x, y: y ** 2 * (x >= 9) * (x < 11) / (-2 * sigma ** 2)
phi3 = lambda x, y: (y - 2) ** 2 * (x >= 9) * (x < 11) * (y >= 0) * (y < 2) / (-2 * sigma ** 2)
phi4 = lambda x, y: ((x - 7) ** 2 + (y - 1) ** 2 - 1) ** 2 / (-2 * sigma ** 2)
phi5 = lambda x, y: (y + x - 8) ** 2 * (x > 7.7) * (x < 8.4) / (-2 * sigma ** 2)
phi6 = lambda x, y: (y - (-0.57 * x + 3.25)) ** 2 * (x > 3.142) * (x < 4.86) / (-2 * sigma ** 2)
phi7 = lambda x, y: ((x - 4) ** 2 + (y - 1) ** 2 - 1) ** 2 * ((y < 0.5) + (y > 1.5)) / (-2 * sigma ** 2)

u = [lambda theta, i=i: np.exp(-np.log(
    np.exp(phi1(theta[0], theta[1])) + np.exp(phi2(theta[0], theta[1])) + np.exp(phi3(theta[0], theta[1])) + np.exp(phi4(theta[0], theta[1])) +
    np.exp(phi5(theta[0], theta[1])) + np.exp(phi6(theta[0], theta[1])) + np.exp(phi7(theta[0], theta[1]))
) / (a ** i)) * (theta[0] <= 13) * (theta[0] >= -2) * (theta[1] >= -1) * (theta[1] <= 3) for i in range(4)]

u1 = [lambda theta, i=i: np.power(
    np.exp(-phi1(theta[0], theta[1])) + np.exp(-phi2(theta[0], theta[1])) + np.exp(-phi3(theta[0], theta[1])) + np.exp(-phi4(theta[0], theta[1])) +
    np.exp(-phi5(theta[0], theta[1])) + np.exp(-phi6(theta[0], theta[1])) + np.exp(-phi7(theta[0], theta[1])),
    -1 / (a ** i)) * ((theta[0] <= 13) * (theta[0] >= -2) * (theta[1] >= -1) * (theta[1] <= 3)) for i in range(4)]


p = [lambda loc: st.multivariate_normal.rvs(mean=loc, cov=np.array([[4, 0], [0, 1]]))] * K
u0 = st.uniform(loc=np.array([-2, -1]), scale=np.array([15, 4])).rvs
x = np.zeros((K, N, 2))
theta0 = np.arange(-2, 13, 0.1)
theta1 = np.arange(-1, 3, 0.1)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
Theta_positions = np.vstack([Theta0.ravel(), Theta1.ravel()])
prob_f = u1[0](Theta_positions)
plt.imshow(prob_f.reshape((150, 40)), interpolation='none')
plt.show()
'''
xs = simple_parallel_tempering(x, K, N, p, us, u0, Ns)
stat = {'acceptance rate': acc / ((N - 1) * K)}
print('acceptance rate: %f' % (stat['acceptance rate']))
fig = plt.figure(figsize=(4, 4))
plt.hist2d(xs[0], xs[1], bins=30)
plt.show()
'''
