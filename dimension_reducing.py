import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3d(data, filename):
    X, y = data
    assert X.shape[0] == 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0,:], X[1,:], X[2,:], c=y)
    plt.savefig(filename)
    plt.show()


def plot2d(data, filename):
    X, y = data
    assert X.shape[0] == 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0,:], X[1,:], c=y)
    plt.savefig(filename)
    plt.show()


def plot1d(data, filename):
    X, y = data
    assert X.shape[0] == 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0,:], X[0,:], c=y)
    plt.savefig(filename)
    plt.show()


def multidimension_scaling(data, new_dim):
    X, y = data
    origal_dim, num_samples = X.shape
    assert new_dim < origal_dim
    dist = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dist[i,j] = np.sum(np.power((X[:,i] - X[:,j]), 2))
    dist2 = np.power(dist, 2)
    dist2_rowm = np.mean(dist2, 1)  # the mean of dist2's rows (feature)
    dist2_colm = np.mean(dist2, 0)  # the mean of dist2's cols (sample)
    dist2_mean = np.sum(dist2) / (num_samples**2)
    b = np.zeros((num_samples, num_samples))  # inner product matrix of Z = dot(W.T, X) b = dot(Z.T, Z)
    for i in range(num_samples):
        for j in range(num_samples):
            b[i,j] = (dist2_rowm[i] + dist2_colm[j] - dist2[i,j] - dist2_mean) / 2
    v, sigma, vt = np.linalg.svd(b)
    Z = np.diag(np.sqrt(sigma)[:new_dim]).dot(vt[:new_dim,:])
    # the eignvector is a column vector
    new_data = Z, y
    return new_data


def pca(data, new_dim):
    X, y = data
    origal_dim, num_samples = X.shape
    x_mean = np.reshape(np.mean(X, axis=1), (-1, 1))
    X = X - x_mean
    x_var = np.reshape(np.mean(np.power(X, 2), axis=1), (-1,1))
    X = X / np.sqrt(x_var)
    x_cov = np.zeros((origal_dim, origal_dim))
    for i in range(num_samples):
        x_cov += X[:,i].dot(X[:,i].T)
    x_cov = x_cov / num_samples
    eigenval, eigenvec = np.linalg.eig(x_cov)
    W = eigenvec[:,:new_dim]
    Z = W.T.dot(X)
    new_data = (Z, y)
    return new_data


if __name__ == '__main__':
    num_samples = 2
    origal_dim = 3
    num_classes = 3
    X = np.random.randn(origal_dim, num_samples)
    y = np.random.randint(num_classes, size=num_samples)
    data = X, y
    plot3d(data, "orginal_data 3d.png")
    new_data = multidimension_scaling(data, new_dim=2)
    plot2d(new_data, "multidimension_scaling 2d.png")
    new_data = multidimension_scaling(new_data, new_dim=1)
    plot1d(new_data, "multidimension_scaling 1d.png")
    new_data = pca(data, 2)
    plot2d(new_data, "pca 2d.png")
    new_data = pca(new_data, 1)
    plot1d(new_data, "pca 1d.png")
