from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
# style.use('fivethirtyeight')
np.random.seed(0)

# from https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
# also see https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137

X = np.linspace(-5, 5, num=20)

number_gaussians_to_generate = 3
X_is = []
for i in range(number_gaussians_to_generate):
    X_i = X * np.random.rand(len(X)) + (i - 1) * 15
    X_is.append(X_i)
X_tot = np.stack(X_is).flatten()  # Combine the clusters to get the random datapoints from above


def plot_data_and_gaussians(p, mu, var, X, iter):
    number_gaussians = len(mu)
    number_examples = len(p)
    plotting_grid = np.linspace(-20, 20, num=60)

    """Plot the data"""
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)

    for i in range(number_examples):
        # determine the color of every datapoint corresponding to the probability it was created by the first up-to-three gaussians
        rgb_color = np.zeros(shape=(3))
        for j in range(number_gaussians):
            rgb_color[j] = p[i][j]

        # normalize the vector to always be a valid color
        rgb_color = rgb_color / np.linalg.norm(rgb_color, ord=1)

        # plot every datapoint according to its p(x | mu, sigma)
        ax0.scatter(X[i], 0.0, color=rgb_color, s=100)

    gaussian_pdfs = []
    for i in range(number_gaussians):
        gaussian = norm(loc=mu[i], scale=var[i])
        gaussian_pdf = gaussian.pdf(plotting_grid)
        gaussian_pdfs.append(gaussian_pdf)

    """Plot the gaussians"""
    for gaussian_pdf, c in zip(gaussian_pdfs, ['r', 'g', 'b']):
        # print(plotting_grid.shape)
        gaussian_pdf = gaussian_pdf.flatten()
        # print(gaussian_pdf.shape)
        ax0.plot(plotting_grid, gaussian_pdf, c=c)
    plt.savefig("new_" + str(iter) + ".png")


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

    """Create the array r with dimensionality n x K"""
    p = np.zeros((number_examples, number_gaussians))

    """
    Probability for each datapoint x_i to belong to gaussian g
    """
    gaussians = []
    for i in range(number_gaussians):
        gaussians.append(norm(loc=mu[i], scale=var[i]))

    for i, gaussian, this_pi in zip(range(number_gaussians), gaussians, pi):
        p[:, i] = this_pi * gaussian.pdf(X)  # Write the probability that x belongs to gaussian i in column i.
        # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians
    """
    Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to
    cluster c
    """
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

    """calculate m_c"""
    total_p_of_gaussians = []
    for c in range(number_gaussians):
        p_this_gaussian_for_all_data = np.sum(p[:, c])
        # For each cluster, calculate the m_c and add it to the list m_c
        total_p_of_gaussians.append(p_this_gaussian_for_all_data)

    """calculate pi_c"""
    for i in range(number_gaussians):
        # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
        pi[i] = (total_p_of_gaussians[i] / total_p)

    """calculate new means mu for each cluster """
    mu = np.sum(x * p, axis=0) / total_p_of_gaussians

    # var_c.append((1/m_c[c]) * np.dot(((np.array(p[:, c]).reshape(60, 1)) *
    #                                 (X.reshape(len(X), 1)-mu[c])).T,
    #                                (X.reshape(len(X), 1)-mu[c])))

    """calculate var_c"""
    var_c = []
    for i in range(number_gaussians):
        ps_this_gaussian = np.array(p[:, i])
        ps_this_gaussian = ps_this_gaussian.reshape(ps_this_gaussian.shape[0], 1)

        distances_points_to_cluster = x - mu[i]
        # distances_points_to_cluster = distances_points_to_cluster / np.max(np.abs(distances_points_to_cluster))

        scaling = (1.0 / total_p_of_gaussians[i])
        # result = scaling * np.dot((this_p * distances_points_to_cluster).T, distances_points_to_cluster)
        sigma_contributions = 0.0
        for j in range(number_examples):
            p_this_gauss_this_example = ps_this_gaussian[j]
            distance_this_example_this_mean = distances_points_to_cluster[j]
            sigma_contributions += p_this_gauss_this_example * distance_this_example_this_mean * distance_this_example_this_mean

        sigma_contributions = sigma_contributions
        sigma_this_cluster = np.sqrt(scaling * sigma_contributions)
        result = sigma_this_cluster
        var_c.append(result)

    return pi, mu, var_c


def em(X_tot, iterations: int = 10):
    mu = [-8, 8, 5]
    pi = [1/3, 1/3, 1/3]
    var = [5, 3, 1]

    mu = [-8, 8]
    pi = [1/3, 1/3]
    var = [5, 3]

    for iter in range(iterations):
        print("Iteration ", iter, " of ", iterations)
        p = e_step(mu=mu, var=var, pi=pi, X=X_tot)
        plot_data_and_gaussians(p=p, mu=mu, var=var, X=X_tot, iter=iter)
        pi, mu, var = m_step(p=p, pi=pi, mu=mu, X=X_tot)


em(X_tot=X_tot, iterations=20)
