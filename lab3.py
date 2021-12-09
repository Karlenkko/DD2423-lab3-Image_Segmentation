import numpy as np
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
        if np.max(np.max(abs(old_centers - centers))) < 0.01:
            break
    print("Converge at " + str(converge + 1))
    segmentation = np.reshape(segmentation, (np.shape(image)[0], np.shape(image)[1]))
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
    # return prob


if __name__ == '__main__':
    print("")
