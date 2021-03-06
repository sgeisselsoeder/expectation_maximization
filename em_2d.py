from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(1234)

# create data set
n = 1000
_mus = np.array([[0, 4], [-2, 0]])
_sigmas = np.array([[[3, 0], [0, 0.5]], [[1, 0], [0, 2]]])
_pis = np.array([0.6, 0.4])
xs = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi*n)) for pi, mu, sigma in zip(_pis, _mus, _sigmas)])

# initial guesses for parameters
pis = np.random.random(2)
pis /= pis.sum()
mus = np.random.random((2, 2))
sigmas = np.array([np.eye(2)] * 2)


def plot_result(pis, mus, sigmas, filename):
    intervals = 101
    ys = np.linspace(-7, 7, intervals)
    X, Y = np.meshgrid(ys, ys)
    _ys = np.vstack([X.ravel(), Y.ravel()]).T

    z = np.zeros(len(_ys))
    for pi, mu, sigma in zip(pis, mus, sigmas):
        z += pi*mvn(mu, sigma).pdf(_ys)
    z = z.reshape((intervals, intervals))

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(111)
    plt.scatter(xs[:, 0], xs[:, 1], alpha=0.2, linewidths=2.0)
    plt.contour(X, Y, z, linewidths=3.0)
    # plt.contour(X, Y, z, N=10)
    # plt.axis([-8, 6, -6, 8])
    ax.axes.set_aspect('equal')
    # plt.tight_layout()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', which='both', labelsize=15)

    plt.savefig(filename)


def em_gmm_orig(xs, pis, mus, sigmas, tol=0.001, max_iter=100):
    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for iter in range(max_iter):
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                ws[j, i] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
        ws /= ws.sum(0)

        # M-step
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]
        pis /= n

        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * xs[i]
            mus[j] /= ws[j, :].sum()

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i] - mus[j], (2, 1))
                sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= ws[j, :].sum()

        # update complete log likelihoood
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

        plot_result(pis=pis, mus=mus, sigmas=sigmas, filename="another_" + str(iter) + ".png")

    return pis, mus, sigmas


def em_gmm_vect(xs, pis, mus, sigmas, tol=0.001, max_iter=100):
    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for iter in range(max_iter):
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(k):
            ws[j, :] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs)
        ws /= ws.sum(0)

        # M-step
        pis = ws.sum(axis=1)
        pis /= n

        mus = np.dot(ws, xs)
        mus /= ws.sum(1)[:, None]

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            ys = xs - mus[j, :]
            sigmas[j] = (ws[j, :, None, None] * np.matmul(ys[:, :, None], ys[:, None, :])).sum(axis=0)
        sigmas /= ws.sum(axis=1)[:, None, None]

        # update complete log likelihood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi*mvn(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

        plot_result(pis=pis, mus=mus, sigmas=sigmas, filename="em2d_" + str(iter + 1) + ".png")

    return pis, mus, sigmas


plot_result(pis=pis, mus=mus, sigmas=sigmas, filename="em2d_0.png")
pis2, mus2, sigmas2 = em_gmm_vect(xs, pis, mus, sigmas)
