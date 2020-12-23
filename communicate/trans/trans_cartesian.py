from compute_energy import pes
from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import torch
import numpy as np
from itertools import combinations
from communicate.get_energy import read_eg

nn = NeuralNetwork('/home/luoshu/PycharmProjects/gaus_ipi/PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (128, 128, 128), 'morse_transform': {'morse': True, 'morse_alpha': 1.3},
          'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'},
          'scale_y': 'std', 'lr': 0.5}

X, y, Xscaler, yscaler = nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('/home/luoshu/PycharmProjects/gaus_ipi/model.pt')


def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1, 3)
    n = len(vec)
    distance_matrix = np.zeros((n, n))
    for i, j in combinations(range(len(vec)), 2):
        R = np.linalg.norm(vec[i] - vec[j])
        distance_matrix[j, i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix), -1)]
    return distance_vector

'''
def do1(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xa + 0.01
    coo1 = [x1, ya, za, xb, yb, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    x2 = xa - 0.01
    coo2 = [x2, ya, za, xb, yb, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_x = (e2 - e1) / 0.02
    y1 = ya + 0.01
    coo1 = [xa, y1, za, xb, yb, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    y2 = ya - 0.01
    coo2 = [xa, y2, za, xb, yb, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_y = (e2 - e1) / 0.02
    z1 = za + 0.01
    coo1 = [xa, ya, z1, xb, yb, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    z2 = za - 0.01
    coo2 = [xa, ya, z2, xb, yb, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_z = (e2 - e1) / 0.02
    return difference_x, difference_y, difference_z


def do2(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xb + 0.01
    coo1 = [xa, ya, za, x1, yb, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    x2 = xb - 0.01
    coo2 = [x2, ya, za, x2, yb, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_x = (e2 - e1) / 0.02
    y1 = yb + 0.01
    coo1 = [xa, y1, za, xb, y1, zb, xc, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    y2 = yb - 0.01
    coo2 = [xa, y2, za, xb, y2, zb, xc, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_y = (e2 - e1) / 0.02
    z1 = zb + 0.01
    coo1 = [xa, ya, za, xb, yb, z1, xc, yc, z1]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    z2 = zb - 0.01
    coo2 = [xa, ya, za, xb, yb, z2, xc, yc, z2]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_z = (e2 - e1) / 0.02
    return difference_x, difference_y, difference_z


def dh1(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xb + 0.01
    coo1 = [xa, ya, za, x1, yb, zb, x1, yc, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    x2 = xc - 0.01
    coo2 = [xa, ya, za, x2, yb, zb, x2, yc, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_x = (e2 - e1) / 0.02
    y1 = yc + 0.01
    coo1 = [xa, ya, za, xb, yb, zb, xc, y1, zc]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    y2 = yc - 0.01
    coo2 = [xa, ya, za, xb, yb, zb, xc, y2, zc]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_y = (e2 - e1) / 0.02
    z1 = zc + 0.01
    coo1 = [xa, ya, za, xb, yb, zb, xc, yc, z1]
    e1 = float(pes(geom_vectors=coo1, cartesian=True))
    z2 = za - 0.01
    coo2 = [xa, ya, za, xb, yb, zb, xc, yc, z2]
    e2 = float(pes(geom_vectors=coo2, cartesian=True))
    difference_z = (e2 - e1) / 0.02
    return difference_x, difference_y, difference_z


def cooO1(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xa + 0.01
    coox1 = [x1, ya, za, xb, yb, zb, xc, yc, zc]
    x2 = xa - 0.01
    coox2 = [x2, ya, za, xb, yb, zb, xc, yc, zc]
    y1 = ya + 0.01
    cooy1 = [xa, y1, za, xb, yb, zb, xc, yc, zc]
    y2 = ya - 0.01
    cooy2 = [xa, y2, za, xb, yb, zb, xc, yc, zc]
    z1 = za + 0.01
    cooz1 = [xa, ya, z1, xb, yb, zb, xc, yc, zc]
    z2 = za - 0.01
    cooz2 = [xa, ya, z2, xb, yb, zb, xc, yc, zc]
    return coox1, coox2, cooy1, cooy2, cooz1, cooz2


def cooO2(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xb + 0.01
    coox1 = [xa, ya, za, x1, yb, zb, xc, yc, zc]
    x2 = xb - 0.01
    coox2 = [xa, ya, za, x2, yb, zb, xc, yc, zc]
    y1 = yb + 0.01
    cooy1 = [xa, ya, za, xb, y1, zb, xc, yc, zc]
    y2 = yb - 0.01
    cooy2 = [xa, ya, za, xb, y2, zb, xc, yc, zc]
    z1 = zb + 0.01
    cooz1 = [xa, ya, za, xb, yb, z1, xc, yc, zc]
    z2 = zb - 0.01
    cooz2 = [xa, ya, za, xb, yb, z2, xc, yc, zc]
    return coox1, coox2, cooy1, cooy2, cooz1, cooz2


def cooH(xa, ya, za, xb, yb, zb, xc, yc, zc):
    x1 = xc + 0.01
    coox1 = [xa, ya, za, xb, yb, zb, x1, yc, zc]
    x2 = xc - 0.01
    coox2 = [xa, ya, za, xb, yb, zb, x2, yc, zc]
    y1 = yc + 0.01
    cooy1 = [xa, ya, za, xb, yb, zb, xc, y1, zc]
    y2 = yc - 0.01
    cooy2 = [xa, ya, za, xb, yb, zb, xc, y2, zc]
    z1 = zc + 0.01
    cooz1 = [xa, ya, za, xb, yb, zb, xc, yc, z1]
    z2 = za - 0.01
    cooz2 = [xa, ya, za, xb, yb, zb, xc, yc, z2]
    return coox1, coox2, cooy1, cooy2, cooz1, cooz2
'''


def geninput():
    # input_value = [0, 0, 1.1125, 0, 0.85, -0.12, 0, 0, 0]
    # print(type(input_value))
    # print(float(pes(geom_vectors=input_value, cartesian=True)))
    final_w = open('/home/luoshu/PycharmProjects/gaus_ipi/communicate/trans/trans.txt', "w")

    # for i in range(len(dirs)):
    f = open('/home/luoshu/PycharmProjects/gaus_ipi/communicate/coordinate.txt', 'r')
    lines = f.readlines()

    m1 = lines[0].split()
    xo1 = float(m1[0])
    yo1 = float(m1[1])
    zo1 = float(m1[2])
    m2 = lines[1].split()
    xo2 = float(m2[0])
    yo2 = float(m2[1])
    zo2 = float(m2[2])
    m3 = lines[2].split()
    xh3 = float(m3[0])
    yh3 = float(m3[1])
    zh3 = float(m3[2])

    # print Cartesian coordinates
    a1 = (xo1, yo1, zo1)
    a2 = (xo2, yo2, zo2)
    a3 = (xh3, yh3, zh3)
    print("The input coordinate for atom 1 is: ", a1)
    print("The input coordinate for atom 2 is: ", a2)
    print("The input coordinate for atom 3 is: ", a3)
    input_value = (xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)
    energy, gradient = pes(geom_vectors=input_value, cartesian=True)

    final_w.write('energy' + '\n')
    final_w.write(str(float(energy.item())) + '\n')
    final_w.write('gradient' + '\n')
    '''
    x1 = do1(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[0]
    y1 = do1(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[1]
    z1 = do1(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[2]
    x2 = do2(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[0]
    y2 = do2(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[1]
    z2 = do2(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[2]
    x3 = dh1(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[0]
    y3 = dh1(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[1]
    z3 = dh1(xo1, yo1, zo1, xo2, yo2, zo2, xh3, yh3, zh3)[2]
    '''
    final_w.write('   1' + '    ' + '8' + '   ' + str(gradient[0]) + ' ' + str(gradient[1]) + ' ' + str(gradient[2]) + '\n')
    final_w.write('   2' + '    ' + '8' + ' ' + str(gradient[3]) + ' ' + str(gradient[4]) + ' ' + str(gradient[5]) + '\n')
    final_w.write('   3' + '    ' + '1' + ' ' + str(gradient[6]) + ' ' + str(gradient[7]) + ' ' + str(gradient[8]) + '\n')

    final_w.close()


geninput()
