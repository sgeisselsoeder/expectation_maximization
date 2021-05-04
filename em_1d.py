from scipy.stats import norm
import numpy as np
import pandas as pd
import sklearn as sk


# from https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
# also see https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137


import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
np.random.seed(0)

X = np.linspace(-5, 5, num=20)
X0 = X*np.random.rand(len(X))+15  # Create data cluster 1
X1 = X*np.random.rand(len(X))-15  # Create data cluster 2
X2 = X*np.random.rand(len(X))  # Create data cluster 3
X_tot = np.stack((X0, X1, X2)).flatten()  # Combine the clusters to get the random datapoints from above


def plot_data_and_gaussians(r, mu, var, X, iter):
    """Plot the data"""
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)

    for i in range(len(r)):
        ax0.scatter(X[i], 0, color=np.array([r[i][0], r[i][1], r[i][2]]), s=100)

    """Plot the gaussians"""
    for g, c in zip([norm(loc=mu[0], scale=var[0]).pdf(np.linspace(-20, 20, num=60)),
                     norm(loc=mu[1], scale=var[1]).pdf(np.linspace(-20, 20, num=60)),
                     norm(loc=mu[2], scale=var[2]).pdf(np.linspace(-20, 20, num=60))], ['r', 'g', 'b']):
        ax0.plot(np.linspace(-20, 20, num=60), g, c=c)
    plt.savefig("new_" + str(iter) + ".png")


def e_step(mu, var, pi, X_tot):
    """Create the array r with dimensionality nxK"""
    r = np.zeros((len(X_tot), 3))

    """
    Probability for each datapoint x_i to belong to gaussian g
    """
    for c, g, p in zip(range(3), [norm(loc=mu[0], scale=var[0]),
                                  norm(loc=mu[1], scale=var[1]),
                                  norm(loc=mu[2], scale=var[2])], pi):
        r[:, c] = p * g.pdf(X_tot)  # Write the probability that x belongs to gaussian c in column c.
        # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians
    """
    Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to
    cluster c
    """
    for i in range(len(r)):
        r[i] = r[i] / (np.sum(pi) * np.sum(r, axis=1)[i])

    return r


def m_step(r, pi, mu, X):
    """M-Step"""

    """calculate m_c"""
    m_c = []
    for c in range(len(r[0])):
        m = np.sum(r[:, c])
        m_c.append(m)  # For each cluster c, calculate the m_c and add it to the list m_c

    """calculate pi_c"""
    for k in range(len(m_c)):
        # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
        pi[k] = (m_c[k]/np.sum(m_c))

    """calculate mu_c"""
    mu = np.sum(X.reshape(len(X), 1)*r, axis=0)/m_c

    # var_c.append((1/m_c[c]) * np.dot(((np.array(r[:, c]).reshape(60, 1)) *
    #                                 (X.reshape(len(X), 1)-mu[c])).T,
    #                                (X.reshape(len(X), 1)-mu[c])))

    """calculate var_c"""
    var_c = []
    for c in range(len(r[0])):
        this_r = np.array(r[:, c])
        # print(this_r.shape)
        this_r = this_r.reshape(this_r.shape[0], 1)

        x = X.reshape(len(X), 1)
        diff = x - mu[c]
        result = (1 / m_c[c]) * np.dot((this_r * diff).T, diff)
        var_c.append(result)

        # var_c.append((1 / m_c[c]) * np.dot((this_r * diff).T, diff))

    return pi, mu, var_c


def em(X_tot, iterations: int = 10):
    mu = [-8, 8, 5]
    pi = [1/3, 1/3, 1/3]
    var = [5, 3, 1]

    for iter in range(iterations):
        r = e_step(mu=mu, var=var, pi=pi, X_tot=X_tot)
        plot_data_and_gaussians(r=r, mu=mu, var=var, X=X_tot, iter=iter)
        pi, mu, _ = m_step(r=r, pi=pi, mu=mu, X=X_tot)


em(X_tot=X_tot, iterations=10)


class GM1D:

    def __init__(self, X, iterations):
        self.iterations = iterations
        self.X = X
        self.mu = None
        self.pi = None
        self.var = None

    def run(self):
        """
        Instantiate the random mu, pi and var
        """
        self.mu = [-8, 8, 5]
        self.pi = [1/3, 1/3, 1/3]
        self.var = [5, 3, 1]
        """
        E-Step
        """
        for iter in range(self.iterations):
            print(self.var)

            """Create the array r with dimensionality nxK"""
            r = np.zeros((len(X_tot), 3))

            """
            Probability for each datapoint x_i to belong to gaussian g
            """
            for c, g, p in zip(range(3), [norm(loc=self.mu[0], scale=self.var[0]),
                                          norm(loc=self.mu[1], scale=self.var[1]),
                                          norm(loc=self.mu[2], scale=self.var[2])], self.pi):
                r[:, c] = p*g.pdf(X_tot)  # Write the probability that x belongs to gaussian c in column c.
                # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians
            """
            Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to
            cluster c
            """
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(self.pi)*np.sum(r, axis=1)[i])

            """Plot the data"""
            fig = plt.figure(figsize=(10, 10))
            ax0 = fig.add_subplot(111)

            for i in range(len(r)):
                ax0.scatter(self.X[i], 0, color=np.array([r[i][0], r[i][1], r[i][2]]), s=100)

            """Plot the gaussians"""
            for g, c in zip([norm(loc=self.mu[0], scale=self.var[0]).pdf(np.linspace(-20, 20, num=60)),
                             norm(loc=self.mu[1], scale=self.var[1]).pdf(np.linspace(-20, 20, num=60)),
                             norm(loc=self.mu[2], scale=self.var[2]).pdf(np.linspace(-20, 20, num=60))], ['r', 'g', 'b']):
                ax0.plot(np.linspace(-20, 20, num=60), g, c=c)

            """M-Step"""

            """calculate m_c"""
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:, c])
                m_c.append(m)  # For each cluster c, calculate the m_c and add it to the list m_c

            """calculate pi_c"""
            for k in range(len(m_c)):
                # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
                self.pi[k] = (m_c[k]/np.sum(m_c))

            """calculate mu_c"""
            self.mu = np.sum(self.X.reshape(len(self.X), 1)*r, axis=0)/m_c

            """calculate var_c"""
            var_c = []

            for c in range(len(r[0])):
                var_c.append((1/m_c[c])*np.dot(((np.array(r[:, c]).reshape(60, 1)) *
                                                (self.X.reshape(len(self.X), 1)-self.mu[c])).T,
                                               (self.X.reshape(len(self.X), 1)-self.mu[c])))

            print(var_c)
            for i in range(len(self.var)):
                self.var[i] = var_c[i][0][0]

            # plt.show()
            plt.savefig("orginal_" + str(iter) + ".png")


GM1D = GM1D(X_tot, 10)
GM1D.run()
