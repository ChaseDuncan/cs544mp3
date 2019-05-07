from utils.preprocess import *

# Step 1: Choose one image per class from the CIFAR-10 dataset of images

data_file = "data/src/cifar-10-batches-py/data_batch_1"
out_file = "data/preprocessed/noisy_images.pkl"
cluster_centers_file = "data/preprocessed/cluster_centers.pkl"
data_dict = unpickle(data_file)

sample_images = sample_dataset(data_dict)

# Step 2: For each image, use k-means to reduce the number of colors to 32

cmap, cluster_centers = colormap(sample_images)
reduced_images = reduce_images(cmap, sample_images)

# Step 3: For each image, prepare a noisy version using the following procedure. 
#   1. For each pixel, sample a Bernoulli random variable with probability 1/32 
#   of coming up 1 (and 31/32 of coming up 0). 
#   2. If your sample has the value 1, then replace the pixel value with a value 
#   chosen uniformly and at random from the range 1-32.

noised_images = add_noise(reduced_images, cluster_centers)      
render_mat(noised_images[0])
store(noised_images, out_file, protocol=2)
store(cluster_centers, cluster_centers_file, protocol=2)

