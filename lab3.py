import numpy as np
import scipy.stats
import sklearn.mixture
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft


def kmeans_segm(image, K, L, seed=42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """

    flatten = np.reshape(image, (-1, 3))
    segmentation = flatten
    np.random.seed(seed)
    # centers = np.random.rand(K, 3) * 255.0
    spectre = np.unique(flatten, axis=0)
    randoms = np.random.randint(low=0, high=np.shape(spectre)[0], size=K)
    centers = spectre[randoms, :]
    old_centers = np.zeros(np.shape(centers))
    # movements = np.zeros(L)
    # for i in range(K):
    #     centers[i, :] = flatten[randoms[i]]
    converge = 0
    err = []
    for i in range(L):
        dists = distance_matrix(flatten, centers)
        # movements[i] = sum(np.min(dists, axis=1))   # pixels
        segmentation = np.argmin(dists,1)   # pixels
        # plt.hist(segmentation)
        # plt.show()
        for center in range(K):
            old_centers[center, :] = centers[center, :]
            cluster_points = np.nonzero(segmentation == center)
            cluster_points = np.reshape(cluster_points, (-1))
            # print(cluster_points)
            mean = np.mean(flatten[cluster_points])
            centers[center, :] = mean
        converge = i
        err.append(np.max(np.max(abs(old_centers - centers))))
        if np.max(np.max(abs(old_centers - centers))) < 0.01:
            break
    print("Converge at " + str(converge + 1))
    # plt.plot(err)
    # plt.title("K=" + str(K))
    # plt.show()
    if len(np.shape(image)) == 3:
        segmentation = np.reshape(segmentation, (np.shape(image)[0], np.shape(image)[1]))
    else:
        segmentation = np.reshape(segmentation, (np.shape(image)[0]))
    return segmentation, centers


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """
    image = image / 255
    Ivec = np.reshape(image, (-1, 3)).astype(np.float32)
    m = np.reshape(mask, (-1))
    masked = Ivec[np.reshape(np.nonzero(m == 1), (-1))]
    size = np.shape(masked)[0]
    segs, centers = kmeans_segm(masked, K, L)
    covariances = []
    p = np.ones((size, K)) * 0.001
    gauss = np.zeros((size, K))
    for i in range(K):
        covariances.append(np.eye(3) * 0.01)

    w = np.zeros(K)
    for i in range(K):
        w[i] = len(np.nonzero(segs == i)) / size

    for i in range(L):
        for j in range(K):
            # print(centers[j])
            gauss[:, j] = w[j] * scipy.stats.multivariate_normal(centers[j], covariances[j]).pdf(masked)

        for j in range(K):
            p[:, j] = np.divide(gauss[:, j], np.sum(gauss, axis=1), where=np.sum(gauss, axis=1)!=0)

        for j in range(K):
            w[j] = np.mean(p[:,j])
            centers[j,:] = np.dot(np.transpose(p[:, j]), masked) / np.sum(p[:, j])
            diff = masked - centers[j,:]
            covariances[j] = np.dot(np.transpose(diff), diff * np.reshape(p[:, j], (-1, 1))) / np.sum(p[:, j])

    # gmm = sklearn.mixture.GaussianMixture(n_components=K, covariance_type="full", max_iter=L)
    # gmm.fit(masked)
    # centers = gmm.means_
    # w = gmm.weights_
    # covariances = gmm.covariances_

    # print(np.shape(prob))

    prob = np.zeros((np.shape(Ivec)[0], K))
    for i in range(K):
        prob[:, i] = w[i] * scipy.stats.multivariate_normal(centers[i], covariances[i]).pdf(Ivec)
        prob[:, i] = prob[:, i] / np.sum(prob[:, i])
    #
    prob = np.sum(prob, axis=1)
    prob = np.reshape(prob, (np.shape(image)[0], np.shape(image)[1]))
    return prob


if __name__ == '__main__':
    print("")
