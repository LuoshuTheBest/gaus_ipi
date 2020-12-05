from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import torch
import numpy as np
from itertools import combinations

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (128, 128, 128), 'morse_transform': {'morse': True, 'morse_alpha': 1.3}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'}, 'scale_y': 'std', 'lr': 0.5}

X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)


# How to use 'compute_energy()' function
# --------------------------------------
# E = compute_energy(geom_vectors, cartesian=bool)
# 'geom_vectors' is either: 
#  1. A list or tuple of coordinates for a single geometry. 
#  2. A column vector of one or more sets of 1d coordinate vectors as a list of lists or 2D NumPy array:
# [[ coord1, coord2, ..., coordn],
#  [ coord1, coord2, ..., coordn],
#      :       :             :  ], 
#  [ coord1, coord2, ..., coordn]]
# In all cases, coordinates should be supplied in the exact same format and exact same order the model was trained on.
# If the coordinates format used to train the model was interatomic distances, each set of coordinates should be a 1d array of either interatom distances or cartesian coordinates. 
# If cartesian coordinates are supplied, cartesian=True should be passed and it will convert them to interatomic distances. 
# The order of coordinates matters. If PES-Learn datasets were used they should be in standard order;
# i.e. cartesians should be supplied in the order x,y,z of most common atoms first, with alphabetical tiebreaker. 
# e.g., C2H3O2 --> H1x H1y H1z H2x H2y H2z H3x H3y H3z C1x C1y C1z C2x C2y C2z O1x O1y O1z O2x O2y O2z
# and interatom distances should be the row-wise order of the lower triangle of the interatom distance matrix, with standard order atom axes:
#    H  H  H  C  C  O  O 
# H 
# H  1
# H  2  3
# C  4  5  6 
# C  7  8  9  10 
# O  11 12 13 14 15
# O  16 17 18 19 20 21

# The returned energy array is a column vector of corresponding energies. Elements can be accessed with E[0,0], E[0,1], E[0,2]
# NOTE: Sending multiple geometries through at once is much faster than a loop of sending single geometries through.


def pes(geom_vectors, cartesian=True):
    model = torch.load('model.pt')
    g = np.asarray(geom_vectors)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = nn.transform_new_X(g, params, Xscaler)
    x = torch.tensor(data=newX, requires_grad=True)
    model.zero_grad()
    # x = torch.tensor(data=x)
    E = model(x.float())
    v = torch.tensor([[1.0]], dtype=torch.float)
    E.backward(v)
    e = nn.inverse_transform_new_y(E.detach(), yscaler)
    # e = e - (insert min energy here)
    # e *= 219474.63  ( convert units )
    return e, x.grad


def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1,3)
    n = len(vec)
    distance_matrix = np.zeros((n,n))
    for i, j in combinations(range(len(vec)),2):
        R = np.linalg.norm(vec[i]-vec[j])
        distance_matrix[j, i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix),-1)]
    return distance_vector


if __name__ == "__main__":
    print("Start the calculation of energy...")
    # Based on the data in PES_data_new 17
    f_w = open('readresult.txt', "w")

#for i in range(len(dirs)):
    f = open('read.txt', 'r')
    lines = f.readlines()

    for index in range(len(lines)):
        m = lines[index].split()
        oo1 = float(m[0])
        oh1 = float(m[1])
        angle = float(m[2])
        print(oo1,oh1,angle)

#print cartisian coordinates
        a1 = (0,0,0)
        a2 = (oo1,0,0)
        degree = angle*np.pi/180
        x = oh1 * np.cos(degree)
        y = oh1 * np.sin(degree)
        a3 = (x,y,0)
     #   f_w.write(str(a1,a2,a3))
        input_value = (0, 0, 0, oo1, 0,0, x, y, 0)
        result, grad = pes(geom_vectors=input_value, cartesian=True)
        f_w.write(str(grad))