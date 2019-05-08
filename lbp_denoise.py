# Had to add because the LBP code is run in python 2.7
from __future__ import division

import numpy as np
import scipy.linalg as la
import pickle
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

c = 1
max_val = 32

def node_str(i, j):
    # Returns node string from coords
    return str(i)+"-"+ str(j)

def coords(node_str):
    # Returns integer coordinates from node name
    split_name  = node_str.split("-")
    return int(split_name[0]), int(split_name[1])

def init_unary(obs, norm=True, smooth=True):
    # Initialize a vector of unary potentials based on the observed value
    diffs = cluster_centers - obs
    diffs = np.square(diffs)
    diffs = np.sum(diffs, axis=1)
    if norm:  
        diffs = diffs / la.norm(diffs)

    if smooth:
        diffs+=1
        diffs = diffs/(len(diffs))

    return diffs.astype(np.float64)

def neighbors(node):
    # Returns the x+1 and y+1 neighbors of nodes encoded as strings
    coord = coords(node)
    return node_str(coord[0]+1, coord[1]), node_str(coord[0], coord[1]+1)

# Make a set of node names where
nodes = []
for i in xrange(32):
    for j in xrange(32):
        nodes.append(node_str(i,j))

g = fg.Graph()
test_img = data[0]

for node in nodes:
    g.rv(node, 32)
    g.factor([node], potential=init_unary(test_img[coords(node)], norm=True))

pairwise = np.ones((max_val, max_val))*c
np.fill_diagonal(pairwise, 0)
for node in nodes:
    x_n, y_n = neighbors(node)
    if x_n in nodes:
        g.factor([node, x_n], potential=pairwise)
        #g.factor([x_n, node], potential=pairwise)
    if y_n in nodes:
        g.factor([node, y_n], potential=pairwise)
        #g.factor([y_n, node], potential=pairwise)

# Run (loopy) belief propagation (LBP)
iters, converged = g.lbp(normalize=True)
#iters, converged = g.lbp()
print('LBP ran for %d iterations. Converged = %r' % (iters, converged))

# Print out the final messages from LBP
g.print_messages()
denoised = np.zeros((32, 32, 3))
for node, marginals in g.rv_marginals():
    import pdb; pdb.set_trace()
    coord = coords(node.name)
    denoised[coord[0], coord[1], :] = cluster_centers[np.argmin(marginals), :]


denoised = denoised.astype(dtype=np.uint8)

render_mat(denoised)

# Print out the final marginals
#g.print_rv_marginals()
