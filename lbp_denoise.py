# Had to add because the LBP code is run in python 2.7
from __future__ import division

import numpy as np
import scipy.linalg as la
import pickle
from sklearn.preprocessing import normalize

from utils.preprocess import *
import factorgraph as fg

data_file = "data/preprocessed/noisy_images.pkl"
cluster_centers_file = "data/preprocessed/cluster_centers.pkl"

data = pickle.load(open(data_file, "rb"))
cluster_centers = np.array(pickle.load(open(cluster_centers_file, "rb")), dtype=np.int)
## Make an empty graph
#g = fg.Graph()
#
## Add some discrete random variables (RVs)
#g.rv('a', 2)
#g.rv('b', 3)
#
## Add some factors, unary and binary
#g.factor(['a'], potential=np.array([0.3, 0.7]))
#g.factor(['b', 'a'], potential=np.array([
#            [0.2, 0.8],
#	    [0.4, 0.6],
#	    [0.1, 0.9],
#	    ]))

c = 0.1
max_val = 32

def node_str(i, j):
    # Returns node string from coords
    return str(i)+"-"+ str(j)

def coords(node_str):
    # Returns integer coordinates from node name
    split_name  = node_str.split("-")
    return int(split_name[0]), int(split_name[1])

def init_unary(obs, norm=True, smooth=True, constant=True):
    # Initialize a vector of unary potentials based on the observed value
    diffs = cluster_centers - obs
    diffs = np.square(diffs)
    diffs = np.sum(diffs, axis=1)

    if constant:
        orig_label_idx = np.argmin(diffs)
        diffs = np.ones(32)
        diffs[orig_label_idx] = 0

    if smooth:
        diffs+=1
        diffs = diffs/(len(diffs))

    if norm:  
        diffs = diffs / la.norm(diffs)

    return 1 - diffs.astype(np.float64)

def neighbors(node):
    # Returns the x+1 and y+1 neighbors of nodes encoded as strings
    coord = coords(node)
    return node_str(coord[0]+1, coord[1]), node_str(coord[0], coord[1]+1)

def denoise(img):
    g = fg.Graph()
    # Make a set of node names where
    nodes = []
    for i in xrange(32):
        for j in xrange(32):
            nodes.append(node_str(i,j))

    for node in nodes:
        g.rv(node, 32)
        g.factor([node], potential=init_unary(img[coords(node)], norm=True))

    pairwise = np.ones((max_val, max_val))*c
    np.fill_diagonal(pairwise, 1)
    pairwise = normalize(pairwise, axis=1)

    for node in nodes:
        x_n, y_n = neighbors(node)
        if x_n in nodes:
            g.factor([node, x_n], potential=pairwise)
        if y_n in nodes:
            g.factor([node, y_n], potential=pairwise)

    # Run (loopy) belief propagation (LBP)
    iters, converged = g.lbp(normalize=True)
    print('LBP ran for %d iterations. Converged = %r' % (iters, converged))

    # Print out the final messages from LBP
    #g.print_messages()
    denoised = np.zeros((32, 32, 3))

    for node, marginals in g.rv_marginals():
        coord = coords(node.name)
        denoised[coord[0], coord[1], :] = cluster_centers[np.argmax(marginals), :]

    return denoised.astype(dtype=np.uint8)

for i in range(len(data)):
    img = data[i]
    render_mat(img, name="pngs/orig/" + str(i) + ".png")
    denoised = denoise(img)
    render_mat(denoised, name="pngs/denoised/" + str(i) + ".png")

