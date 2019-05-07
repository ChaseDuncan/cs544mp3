
# Had to add because the LBP code is run in python 2.7
from __future__ import division

import pickle
import numpy as np
import copy
from PIL import Image

from sklearn.cluster import KMeans
from collections import defaultdict
from numpy.random import binomial, randint

# STRUCTURE OF DATA
#
# Each data_batch_* file encodes a dictionary with 2 keys:
#
# 'data' -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
# The first 1024 entries contain the red channel values, the next 1024 the green, and the final 
# 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array 
# are the red channel values of the first row of the image.

# 'labels' -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of 
# the ith image in the array data.

def store(data, file, protocol=3):
    # Pickles data. 
    with open(file, 'wb') as fo: 
        pickle.dump(data, fo, protocol=protocol) 

def unpickle(file):
    # Unpickle a CIFAR 10 split 

    with open(file, 'rb') as fo: 
	try:
	    samples = pickle.load(fo)
	except UnicodeDecodeError:  #python 3.x
	    fo.seek(0)
	    samples = pickle.load(fo, encoding='latin1')
    return samples

def slice_by_color(vectorized_img):
    # Splits the vectorized image into component color vectors
        r = vectorized_img[:1024]
        g = vectorized_img[1024:2*1024]
        b = vectorized_img[2* 1024:]
        return r, g, b

def vec2mat(vectorized_img):
    # Convert vectorized image data to RGB tensor
    r, g, b = slice_by_color(vectorized_img)
    img_dims = (32, 32)
    return np.stack([r.reshape(img_dims), g.reshape(img_dims), b.reshape(img_dims)], axis=-1)

def render(vectorized_img):
    # Render image for debugging purposes.
    img_mat = vec2mat(vectorized_img)
    render_mat(img_mat)

def render_mat(img_mat):
    img = Image.fromarray(img_mat, 'RGB')
    img.save("blah.png")
    img.show()

def create_cluster_examples(sample_images):
    # Convert images in into 1x3 vectors of RGB pixel values for clustering.
    data_pts = []
    for img in sample_images.values():
        r, g, b = slice_by_color(img)
        data_pts.extend([np.array((rv, gv, bv)) for rv, gv, bv in zip(r, g, b)])
    return data_pts

def sample_dataset(data_dict):
    # Extract an image for each label from CIFAR batch
    sample_images = defaultdict()

    for img, label in zip(data_dict['data'], data_dict['labels']):
        if label in sample_images.keys():
            continue
        sample_images[label] = img
        if len(sample_images) == 10:
            break 
    return sample_images

def colormap(sample_images):
    # Generates colormap which maps a tuple of coords in orig color space to reduced space
    X = create_cluster_examples(sample_images)
    kmeans = KMeans(n_clusters=32, max_iter=1000, random_state=0).fit(X)

    colormap = defaultdict()
    cluster_centers = []
    for v in kmeans.cluster_centers_:
        cluster_centers.append(v.astype(dtype=np.uint8))

    for coord, label in zip(X, kmeans.labels_):
        colormap[tuple(coord)] = cluster_centers[label]
    return colormap, cluster_centers

def reduce_images(colormap, sample_images):
    # Reduces the dimensionality of the color space of the ima
    reduced_images = []
    for image in sample_images.values():
        reduced_image = []
        rv, gv, bv = slice_by_color(image)
        for (r, g, b) in zip(rv, gv, bv):
            trans_coords = colormap[(r, g, b)]
            reduced_image.append(trans_coords)
        
        reduced_images.append(np.ravel(np.array(reduced_image), order='F'))

    return reduced_images

def add_noise(reduced_images, cluster_centers, p=1/32, low=0, high=32):
    # Randomly adds noise to images.
    noised_images = copy.deepcopy(reduced_images[:])
    noised_images = [vec2mat(image) for image in noised_images]
    #cluster_centers = np.array(cluster_centers)
    num_pixels = noised_images[0].shape[0] 
    for i in range(len(noised_images)):
        s = binomial(1, p, (num_pixels, num_pixels))
        np.set_printoptions(threshold=np.inf)
        num_noise = len(np.where(s>0)[1])

        noised_images[i][np.where(s>0)] = \
                np.array([cluster_centers[randint(low, high)] for j in range(num_noise)]) #[low, high)
    return noised_images

