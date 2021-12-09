import sys
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from lab3 import kmeans_segm
from Functions import mean_segments, overlay_bounds

def kmeans_example():
    K = 10              # number of clusters used
    L = 50              # number of iterations
    seed = 14           # seed used for random initialization
    scale_factor = 0.5  # image downscale factor
    image_sigma = 1.0   # image preblurring scale
    
    img = Image.open('Images-jpg/orange.jpg')
    # img = Image.open('Images-jpg/tiger1.jpg')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))
    
    h = ImageFilter.GaussianBlur(image_sigma)
    I = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)
    segm, centers = kmeans_segm(I, K, L, seed)
    Iseg = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    img = Image.fromarray(Inew.astype(np.ubyte))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(Image.fromarray(Iseg.astype(np.ubyte)))
    plt.axis('off')
    plt.title("K=" + str(K))
    plt.show()
    img.save('result/kmeans.png')

if __name__ == '__main__':
    sys.exit(kmeans_example())
