from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import torch
import numpy as np
from itertools import combinations
import os
from peslearn.constants import package_directory
import re

nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (128, 128, 128), 'morse_transform': {'morse': False},
          'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'},
          'scale_y': 'std', 'lr': 0.6}

# Why all r1 in the input data is 0.97? I am very confused.
X, y, Xscaler, yscaler = nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model.pt')

# Outline of backpropagation algorithm development:
# 1. Implementation of different components of backpropagation
# 2. Testing of the components
# 3. Testing of the data transfer between components

# Testing procedure for backpropagation algorithm:
# 1. The conversion of coordinate (from Cartesian to distance): implemented and checked
# 2. The Morse transform: implemented and checked
# 3. The fundamental -> invariant: implemented and unchecked
# 4. Degree reduction: implemented and unchecked
# 5. The scaling of X: implemented and checked
# 6. The gradient of NN: implemented and unchecked
# 7. The scaling of Y: implemented and checked

# Procedure of joint testing:
# 1. dE / d(y_scale)
# 2. dE / d(NN)
# 3. dE / d(x_scale)
# 4. dE / d(reduce)
# 5. dE / d(inter-atomic)
# 6. dE / d(morse)
# 7. dE / d(xyz)


def pes(geom_vectors, cartesian=True):
    model = torch.load('model.pt')
    g = np.asarray(geom_vectors)
    gradients_wrt_cartesian_coordinates = np.full((3, len(g)//3), 1, dtype=float)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        gradients_wrt_cartesian_coordinates = np.apply_along_axis(cart_conversion_gradient, axis, g)
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = nn.transform_new_X(g, params, Xscaler)
    morse_grad_array = np.full((3, ), 1, dtype=float)
    g_after_morse = g
    if params["morse_transform"]["morse"]:
        morse_grad, g_after_morse = transform_x_buffer(g, params)
        morse_grad_array = np.array(morse_grad)
    # Compute the gradient of inter-atomics to fundamental and degree reduction
    pip_grad_array = np.full((3, 3), 1, dtype=float)
    degree_reduction_grad_array = np.full((3, ), 1, dtype=float)
    if params["pip"]["pip"]:
        target_path = os.path.join(package_directory, "lib", nn.molecule_type, "output")
        x_after_pip, pip_grad_array, degrees = get_gradient_from_interatomics_to_fundamental(g_after_morse, target_path)
        if params["pip"]["degree_reduction"]:
            degree_reduction_grad_array = get_gradient_from_degree_reduction(x_after_pip, degrees)
    joint_grad_pip_array = pip_grad_array.dot(degree_reduction_grad_array)
    print("The gradient computed by morse preprocessing: ", morse_grad_array)
    # Divide the gradient computation into different sub-categories
    x_scale = Xscaler.scale_
    # Because the scale returned by the std scalar is reversed (input wrt output), need to reverse the result to
    # balance.
    if "mm" not in params["scale_X"]["scale_X"]:
        x_scale = 1 / x_scale
    # On the other hand, the scale of MinMaxScaler is not reversed (output wrt input), no need to reverse.
    concatenated_gradient = []
    for i in range(len(x_scale)):
        x_scale_element = x_scale[i]
        morse_grad_element = morse_grad_array[i]
        joint = joint_grad_pip_array[i]
        concatenated_gradient.append(x_scale_element * morse_grad_element * joint)
    concatenated_gradient_array = np.array(concatenated_gradient)
    x = torch.tensor(data=newX, requires_grad=True)
    model.zero_grad()
    E = model(x.float())
    E.backward()
    print("Gradient of x: ", x.grad)
    # Start from the computation of gradient with the scaling of y
    # Change the file to the pytorch code
    energy_to_compute = E.detach()
    e = nn.inverse_transform_new_y(energy_to_compute, yscaler)
    # The scale is reversed after reversion. No need to reverse for y scaling.
    y_scale = yscaler.scale_
    if "mm" in params["scale_y"]:
        y_scale = 1 / y_scale
    nn_grad = x.grad
    nn_grad.require_grad_ = False
    print("Before gradient: ", nn_grad)
    main_grad = np.array(nn_grad * y_scale)
    print("Main grad: ", main_grad[0])
    final_grad = []
    for i in range(len(main_grad[0])):
        grad_element = main_grad[0][i] * concatenated_gradient_array[i]
        final_grad.append(grad_element)
    final_result = gradients_wrt_cartesian_coordinates.dot(np.array(final_grad))
    return e, final_result


def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1, 3)
    n = len(vec)
    distance_matrix = np.zeros((n, n))
    for i, j in combinations(range(len(vec)), 2):
        R = np.linalg.norm(vec[i] - vec[j])
        distance_matrix[j, i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix), -1)]
    return distance_vector


# This function copies the function of conversion of cartesian coordinate to the distance for the purpose of gradient
# computation.
def cart_conversion_gradient(coordinates):
    gradients = []
    for i in range(len(coordinates)//3):
        gradients.append(get_gradient_from_cart_conversion(coordinates, i))
    gradient_matrix = np.array(gradients)
    return gradient_matrix.transpose()


# A helper function of computing the gradient of coordinate conversion.
def get_gradient_from_cart_conversion(coordinates_file, target_axis):
    coordinates_vector = torch.tensor(coordinates_file, requires_grad=True)
    coordinates_matrix = torch.reshape(coordinates_vector, (len(coordinates_vector)//3, 3))
    internal_coordinates_matrix = torch.zeros((len(coordinates_matrix), len(coordinates_matrix)))
    for i, j in combinations(range(len(coordinates_matrix)), 2):
        norm = torch.norm(coordinates_matrix[i] - coordinates_matrix[j])
        internal_coordinates_matrix[j, i] = norm
    internal_coordinates_vector = internal_coordinates_matrix[np.tril_indices(len(internal_coordinates_matrix), -1)]
    internal_coordinates_vector[target_axis].backward()
    target_gradient = np.array(list(np.array(coordinates_vector.grad)))
    return target_gradient


# This function copies the original preprocessing function for the purpose of gradient computation.
def transform_x_buffer(cartesian_coordinate, parameters):
    x_tensor = torch.tensor(cartesian_coordinate, requires_grad=True)
    if len(cartesian_coordinate.shape) > 2:
        raise ValueError("The input dimension goes beyond the valid region.")
    x_tensor_after_transformation = x_tensor
    if parameters["morse_transform"]["morse"]:
        alpha = parameters["morse_transform"]["morse_alpha"]
        x_tensor_after_transformation = torch.exp(-x_tensor / alpha)
    x_sum = x_tensor_after_transformation.sum()
    x_sum.backward()
    return x_tensor.grad, np.array(x_tensor_after_transformation)


# This function copies the original function of converting the inter-atomics to fundamental
def get_gradient_from_interatomics_to_fundamental(raw_X, fi_path):
    if len(raw_X.shape) == 1:
        raw_X = np.expand_dims(raw_X, 0)
    nbonds = raw_X.shape[1]
    with open(fi_path, 'r') as f:
        data = f.read()
        data = re.sub('\^', '**', data)
        #  convert subscripts of bonds to 0 indexing
        for i in range(1, nbonds + 1):
            data = re.sub('x{}(\D)'.format(str(i)), 'x{}\\1'.format(i - 1), data)
        polys = re.findall("\]=(.+)", data)
    # create a new_X matrix that is the shape of number geoms, number of Fundamental Invariants
    new_X = np.zeros((raw_X.shape[0], len(polys)))
    conversion_grad_list = []
    for i, p in enumerate(polys):  # evaluate each FI
        raw_x_tensor = torch.tensor(raw_X, requires_grad=True)
        new_x_tensor = torch.tensor(new_X)
        # convert the FI to a python expression of raw_X, e.g. x1 + x2 becomes raw_X[:,1] + raw_X[:,2]
        eval_string = re.sub(r"(x)(\d+)", r"raw_x_tensor[:,\2]", p)
        eval_string_array = re.sub(r"(x)(\d+)", r"raw_X[:,\2]", p)
        # evaluate that column's FI from columns of raw_X
        new_X[:, i] = eval(eval_string_array)
        new_x_tensor[:, i] = eval(eval_string)
        new_x_element = new_x_tensor[0][i]
        new_x_element.backward(retain_graph=True)
        conversion_grad_list.append(np.array(raw_x_tensor.grad[0].detach()))
    # conversion_gradient = np.squeeze(np.array(raw_x_tensor.grad))
    # find degree of each FI
    degrees = []
    for p in polys:
        # just checking first, assumes every term in each FI polynomial has the same degree (seems to always be true)
        tmp = p.split('+')[0]
        # count number of exponents and number of occurances of character 'x'
        exps = [int(i) - 1 for i in re.findall("\*\*(\d+)", tmp)]
        ndegrees = len(re.findall("x", tmp)) + sum(exps)
        degrees.append(ndegrees)
    return np.squeeze(new_X), np.array(conversion_grad_list).transpose(), degrees


def get_gradient_from_degree_reduction(raw_X, degrees):
    if len(raw_X.shape) == 1:
        raw_X = np.expand_dims(raw_X, 0)
    x_tensor = torch.tensor(raw_X, requires_grad=True)
    new_x_tensor = torch.tensor(raw_X)
    for i, degree in enumerate(degrees):
        new_x_tensor[:, i] = torch.pow(x_tensor[:, i], 1/degree)
        raw_X[:, i] = np.power(raw_X[:, i], 1 / degree)
    new_x_tensor.sum().backward()
    grad_degree_reduction = x_tensor.grad
    grad_degree_reduction_array = np.squeeze(np.array(grad_degree_reduction))
    return grad_degree_reduction_array


# Changed to new branch luoshu_implementation
if __name__ == "__main__":
    print("Start the calculation of energy...")
    # Based on the data in PES_data_new 17
    input_value = (0, 0, 1.1125, 0, 0.85, -0.12, 0, 0, 0)
    input_value_plus = (0, 0, 1.1125, 0, 0.85, -0.12, 0, 0, 0 + 1e-2)
    input_value_minus = (0, 0, 1.1125, 0, 0.85, -0.12 - 1e-2, 0, 0, 0)
    result, grad = pes(geom_vectors=input_value, cartesian=True)
    result_plus, _ = pes(geom_vectors=input_value_plus, cartesian=True)
    result_minus, _ = pes(geom_vectors=input_value_plus, cartesian=True)
    grad_test = (result_plus - result) / 1e-2
    print("Computed energy: ", result)
    print("Derived gradient: ", grad)
