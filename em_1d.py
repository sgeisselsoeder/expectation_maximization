from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
# style.use('fivethirtyeight')
np.random.seed(0)

# from https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
# also see https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137

datenpunkte_pro_cluster = 20
clusterbereich = np.linspace(-5, 5, num=datenpunkte_pro_cluster)

number_clusters_to_generate = 3
clusters = []
for i in range(number_clusters_to_generate):
    # Datenpunkte, jeweils auf der x-Achse verschoben
    cluster = clusterbereich * np.random.rand(datenpunkte_pro_cluster) + (i - 1) * 18
    clusters.append(cluster)
# Die Beispieldaten X bestehen aus der Summe aler Cluster
X = np.stack(clusters).flatten()


def plot_setup(X):
    number_examples = len(X)

    # Eine Figure zum darstellen der Daten erstellen
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.set_xlim([-33, 33])
    ax0.set_ylim([-0.02, 0.4])

    # Die Datenpunkte plotten
    rgb_color = np.array([0.0, 0.0, 1.0])
    for i in range(number_examples):
        ax0.scatter(X[i], 0.0, color=rgb_color, s=100)

    ax0.set_xlabel("x")
    ax0.set_yticks([])
    ax0.set_yticks([], minor=True)
    ax0.xaxis.label.set_size(30)
    ax0.tick_params(axis='both', which='both', labelsize=20)

    plt.savefig("em1d_0.png")
    plt.close()


def plot_data_and_gaussians(p, mu, var, X, iter):
    number_gaussians = len(mu)
    number_examples = len(p)

    # Eine Figure zum darstellen der Daten erstellen
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.set_ylim([-0.02, 0.4])

    # Die Datenpunkte plotten
    rgb_color = np.zeros(shape=(3))
    for i in range(number_examples):
        # Die Farbe der Datenpunkte entspricht der Verteilung der Wahrscheinlichkeiten p(x | mu, sigma)
        # (wie wahrscheinlich welcher Gauss den Datenpunkt erzeugt und nur für die ersten drei).
        for j in range(min(number_gaussians, 3)):
            rgb_color[j] = min(p[i][j], 1.0)

        # Normieren der Farben ist notwendig für die Darstellung
        rgb_color = rgb_color / np.linalg.norm(rgb_color, ord=1)

        # Plotten der Datenpunkte
        ax0.scatter(X[i], 0.0, color=rgb_color, s=100)

    # Ploten der Gaussverteilungen
    plotting_grid = np.linspace(-30, 30, num=300)
    colors = ['r', 'g', 'b', 'm']
    color = colors[0]
    for i in range(number_gaussians):
        gaussian = norm(loc=mu[i], scale=var[i])
        gaussian_pdf = gaussian.pdf(plotting_grid).flatten()

        if i < len(colors):
            color = colors[i]
        ax0.plot(plotting_grid, gaussian_pdf, c=color, linewidth=3.0)

    ax0.set_xlabel("x")
    ax0.set_yticks([])
    ax0.set_yticks([], minor=True)
    ax0.xaxis.label.set_size(30)
    ax0.tick_params(axis='both', which='both', labelsize=20)

    plt.savefig("em1d_" + str(iter + 1) + ".png")
    plt.close()


# y(z) = pi N(x, mu, Sigma) / sum(N(x, mu, Sigma))
# y = f(pi, x, mu, Sigma)
# y = f(mu, var, pi, x)
# y = p
def e_step(mu, var, pi, X):
    """ Calculates probabilities p for every datapoint how likely it has been generated by every gaussian.

    Args:
    mu: Means of the Gaussian distributions.
    var: Variances of the Gaussian distributions.
    pi: Scaling factors ("height") for the Gaussian distributions.
    X: The 1D data in the format X[index_of_example], with every x = X[i] being a scalar

    Returns:
    The probabilities p
    """

    number_gaussians = len(mu)
    number_examples = len(X)
    p = np.zeros((number_examples, number_gaussians))

    # Probability for each datapoint x_i to belong to gaussian g
    gaussians = []
    for i in range(number_gaussians):
        gaussians.append(norm(loc=mu[i], scale=np.sqrt(var[i])))

    for i, gaussian, this_pi in zip(range(number_gaussians), gaussians, pi):
        p[:, i] = this_pi * gaussian.pdf(X)  # Write the probability that x belongs to gaussian i in column i.

    # Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to cluster c
    for i in range(len(p)):
        p[i] = p[i] / (np.sum(pi) * np.sum(p, axis=1)[i])

    return p


# N_k = total_p_of_gaussians[i]
def m_step(p, pi, mu, X):
    """M-Step"""
    number_gaussians = len(mu)
    number_examples = len(X)
    x = X.reshape(number_examples, 1)

    total_p = np.sum(p)

    """calculate "contribution weights" of gaussians"""
    total_p_of_gaussians = []
    for c in range(number_gaussians):
        p_this_gaussian_for_all_data = np.sum(p[:, c])
        total_p_of_gaussians.append(p_this_gaussian_for_all_data)

    """update the fraction of contributions (pi) from each cluster"""
    for i in range(number_gaussians):
        pi[i] = (total_p_of_gaussians[i] / total_p)

    """update means mu for each cluster """
    mu = np.sum(p * x, axis=0) / total_p_of_gaussians

    """update variances var"""
    var_c = []
    for i in range(number_gaussians):
        ps_this_gaussian = np.array(p[:, i])
        ps_this_gaussian = ps_this_gaussian.reshape(ps_this_gaussian.shape[0], 1)

        distances_examples_to_cluster = x - mu[i]
        distances_squared = distances_examples_to_cluster * distances_examples_to_cluster

        sigma_this_cluster = (1.0 / total_p_of_gaussians[i]) * np.sum(ps_this_gaussian * distances_squared)

        var_c.append(sigma_this_cluster)

    return pi, mu, var_c


def em(X, iterations: int = 10):
    mu = [-8, 8, 5]
    pi = [1/3, 1/3, 1/3]
    var = [5, 3, 1]

    plot_setup(X=X)
    for iter in range(iterations):
        print("Iteration ", iter, " of ", iterations)
        p = e_step(mu=mu, var=var, pi=pi, X=X)
        plot_data_and_gaussians(p=p, mu=mu, var=var, X=X, iter=iter)
        pi, mu, var = m_step(p=p, pi=pi, mu=mu, X=X)


em(X=X, iterations=14)
