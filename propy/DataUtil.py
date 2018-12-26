import numpy as np


def ones_feature(num_nodes, num_features):
    return np.ones(shape=(num_nodes, num_features))


def matrix_to_list(matrix, default_value=0):
    lst = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val != default_value:
                lst.append([i, j, val])
    return lst


def list_to_matrix(lst, size, default_value=0):
    matrix = np.full(shape=(size, size), fill_value=default_value)
    for i, j, val in lst:
        matrix[i][j] = val
    return matrix
