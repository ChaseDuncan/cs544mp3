import pickle
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def unpickle(file):
    # Unpickle a CIFAR 10 split
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

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
    from PIL import Image
    img_mat = vec2mat(vectorized_img)
    img = Image.fromarray(img_mat, 'RGB')
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
    return colormap


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


# Step 1: Choose one image per class from the CIFAR-10 dataset of images

data_file = "data/src/cifar-10-batches-py/data_batch_1"
data_dict = unpickle(data_file)

sample_images = sample_dataset(data_dict)

#  Step 2: For each image, use k-means to reduce the number of colors to 32

cmap = colormap(sample_images)

def reduce_images(colormap, sample_images):
    reduced_images = []
    for image in sample_images.values():
        reduced_image = []
        rv, gv, bv = slice_by_color(image)
        for (r, g, b) in zip(rv, gv, bv):
            trans_coords = cmap[(r, g, b)]
            reduced_image.append(trans_coords)
        
        #import pdb; pdb.set_trace()
        reduced_images.append(np.ravel(np.array(reduced_image)))

    return reduced_images

reduced_images = reduce_images(cmap, sample_images)

render(reduced_images[2])

