import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot2d(data, filename):
    X, y = data
    assert X.shape[0] == 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0,:], X[1,:], c=y)
    plt.savefig(filename)
    plt.show()

def plot3d(data, filename):
    X, y = data
    assert X.shape[0] == 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0,:], X[1,:], X[2,:], c=y)
    plt.savefig(filename)
    plt.show()

def kMeans(data, k):
    x = data
    features, samples = x.shape
    assert k <= samples
    mu = np.zeros((features, k))
    c = np.zeros(samples)
    init_mu = np.random.randint(samples, size=k)
    mu = x[:,init_mu]
    iters = 5
    for iter in range(iters):
        plot3d((x,c),'iter_{}.png'.format(iter))
        for i in range(samples):
            dist_xi_mu = np.sum(np.power((x[:,i].reshape((-1,1)) - mu), 2), axis=0)
            c[i] = np.argmin(dist_xi_mu)
        for j in range(k):
            labels = c == j
            mu_j = np.sum(labels * x, axis=1) / np.sum(labels)



if __name__ == '__main__':
    origal_dim = 3
    num_samples = 1000
    X = np.random.randn(origal_dim, num_samples)
    cluster centroids = 5
    kMeans(X, cluster centroids)
