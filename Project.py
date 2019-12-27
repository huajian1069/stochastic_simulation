import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def metropolis_hastings(x, p, u):
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


def simple_parallel_tempering(x, N, p, u, u0, Ns):
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


def random_walk_metropolis(N, p, u, u0):
    x = np.zeros((N,))
    x[0] = u0()
    for n in range(N - 1):
        x[n + 1] = metropolis_hastings(x[n], p, u)
    return x


# ex2
acc = 0
K = 4
N = 10000
p = [lambda loc: st.norm.rvs(loc=loc, scale=3)] * K
gammas = [1, 2, 4, 8, 16]
T = 2
u0 = st.uniform(loc=-3, scale=6).rvs
Ns = 1
x = np.zeros((K, N))
x_t = np.linspace(-5, 5, 1000)
fig = plt.figure(figsize=(20, 4))
for j, gamma in enumerate(gammas):
    u = [lambda x, i=i: np.exp(-gamma * (x ** 2 - 1) ** 2 / (T ** i)) for i in range(4)]
    acc = 0
    xs = simple_parallel_tempering(x, N, p, u, u0, Ns)
    xs_walk = random_walk_metropolis(N, p[0], u[0], u0)
    stat = {'acceptance rate': acc / ((N - 1) * K)}
    print('acceptance rate when gamma=%d: %f' % (gamma, stat['acceptance rate']))
    y_t = u[0](x_t)

    ax = fig.add_subplot(2, 5, j + 1)
    ax.hist(xs_walk, bins=100, density=True, label='Walk')
    ax.hist(xs, bins=100, density=True, label='PT')
    ax.plot(x_t, 0.22 * y_t / stat['acceptance rate'], linewidth=1., label=r'$\tilde{f}(x)$')
    ax.set_title('gamma = ' + str(gamma))

    ax2 = fig.add_subplot(2, 5, 5 + j + 1)
    ax2.hist(xs_walk, bins=100, density=True, label='Walk')
    ax2.plot(x_t, 0.22 * y_t / stat['acceptance rate'], linewidth=1., label=r'$\tilde{f}(x)$')
    ax2.set_title('gamma = ' + str(gamma))

plt.legend()
# plt.savefig('figures/project_hist.png')
plt.show()

# ex3
sigma = 0.05
phi1 = lambda x, y: ((x - 1.7) ** 2 + (y - 1) ** 2 - 1) ** 2 * (x <= 2.5) / (-2 * sigma ** 2)
phi2 = lambda x, y: y ** 2 * (x >= 9) * (x < 11) / (-2 * sigma ** 2)
phi3 = lambda x, y: (y - 2) ** 2 * (x >= 9) * (x < 11) * (y >= 0) * (y < 2) / (-2 * sigma ** 2)
phi4 = lambda x, y: ((x - 7) ** 2 + (y - 1) ** 2 - 1) ** 2 / (-2 * sigma ** 2)
phi5 = lambda x, y: (y + x - 8) ** 2 * (x > 7.7) * (x < 8.4) / (-2 * sigma ** 2)
phi6 = lambda x, y: (y - (-0.57 * x + 3.25)) ** 2 * (x > 3.142) * (x < 4.86) / (-2 * sigma ** 2)
phi7 = lambda x, y: ((x - 4) ** 2 + (y - 1) ** 2 - 1) ** 2 * ((y < 0.5) + (y > 1.5)) / (-2 * sigma ** 2)

u = [lambda x, y, i=i: np.exp(-np.log(
    np.exp(-phi1(x, y)) + np.exp(-phi2(x, y)) + np.exp(-phi3(x, y)) + np.exp(-phi4(x, y)) +
    np.exp(-phi5(x, y)) + np.exp(-phi6(x, y)) + np.exp(-phi7(x, y))
) * (x >= -2) * (x <= 13) * (y >= -1) * (y <= 3) / (T ** i)) for i in range(4)]
us = [lambda xy, i=i: u[i](xy[0], xy[1]) for i in range(4)]
p = [lambda loc: st.multivariate_normal.rvs(mean=loc, cov=np.array([[4, 0], [0, 1]]))] * K
u0 = st.uniform(loc=np.array([-2, -1]), scale=np.array([15, 4])).rvs
x = np.zeros((K, N, 2))
acc = 0
N = 100
xs = simple_parallel_tempering(x, N, p, us, u0, Ns)
stat = {'acceptance rate': acc / ((N - 1) * K)}
print('acceptance rate: %f' % (stat['acceptance rate']))
fig = plt.figure(figsize=(4, 4))
plt.hist2d(xs[0], xs[1], bins=30)
plt.show()
