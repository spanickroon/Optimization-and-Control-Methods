import numpy as np
import time
m_a = [
    '0 1 4 1 0 -3 5 0',
    '1 -1 0 1 0 0 1 0',
    '0 7 -1 0 -1 3 8 0',
    '1 1 1 1 0 3 -3 1'
]

v_b = '6 10 -2 15'
v_c = '-5 -2 3 -4 -6 0 -1 -5'
v_x = '4 0 0 6 2 0 0 5'
v_jb = '1 4 5 8'


m_a = [
    '0 1 4 1 0 -8 1 5',
    '0 -1 0 -1 0 0 0 0',
    '0 2 -1 0 -1 3 -1 0',
    '1 1 1 1 0 3 1 1'
]

v_b = '36 -11 10 20'
v_c = '-5 2 3 -4 -6 0 1 -5'
v_x = '4 5 0 6 0 0 0 5'
v_jb = '1 2 4 8'


m_a = [
    '1 2 1 0 0 0',
    '2 1 0 1 0 0',
    '1 0 0 0 1 0',
    '0 1 0 0 0 1'
]

v_b = '10 11 5 4'
v_c = '20 26 0 0 0 1'
v_x = '2 4 0 3 3 0'
v_jb = '5 2 1 4'

m_a = [
    '1 1',
]

v_b = '1'
v_c = '1 1'
v_x = '1 1'
v_jb = '1'


def ban_linalg_inv(source_matrix):
    size = len(source_matrix)
    matrix = np.concatenate((source_matrix, np.eye(size)), axis=1)

    for i in range(size):
        j = i
        while matrix[j][i] == 0 and j < size:
            j += 1
        if j == size:
            return None
        if j > 0:
            matrix[[i, j]] = matrix[[j, i]]

        matrix[i] = matrix[i] / matrix[i][i]

        for j in range(i + 1, size):
            matrix[j] -= matrix[i] * matrix[j][i] * matrix[i][i]

    for i in range(size-1, -1, -1):
        for j in range(i-1, -1, -1):
            matrix[j] = matrix[j] - matrix[j][i] * matrix[i]

    return matrix[:, size:]


def input_matrix(m):
    matrix = []
    # range(m)
    for el in m_a:
        matrix.append(list(map(int, el.split())))
    return np.array(matrix)


def input_vector(v):
    return list(map(int, v.split()))


def input_float_vector(v):
    return list(map(float, v.split()))


def create_matrix_ab(matrix_a, vector_jb):
    size = len(vector_jb)
    matrix_ab = np.zeros((size, size))

    for index, el in enumerate(vector_jb):
        matrix_ab[:, index] = matrix_a[:, el-1]

    return matrix_ab


def create_vector_cb(vector_c, vector_jb):
    vector_cb = np.zeros(len(vector_jb))

    for i, el in enumerate(vector_jb):
        vector_cb[i] = vector_c[el-1]

    return vector_cb


def find_min_delta(vector_delta, all_j, vector_jb):
    vector_jb_H = all_j - set(vector_jb)

    if vector_jb_H == set():
        return (1, 1)

    min_delta = []
    for i in vector_jb_H:
        min_delta.append(vector_delta[i-1])

    delta = min(min_delta)
    return (min_delta.index(delta), delta)


def min_theta(vector_x, vector_z, vector_jb):
    vector_theta = []
    min_index = 0
    min_value = -1
    i = 0
    for z, jb in zip(vector_z, vector_jb):
        if z > 0:
            theta = vector_x[jb-1] / z
            vector_theta.append(theta)
            if theta <= min(vector_theta):
                min_index = i
                min_value = theta
        i += 1
    return (min_value, min_index+1)


def create_vector_x_new(vector_x, vector_z, vector_jb, j0, theta, m):
    vector_x_new = np.zeros_like(vector_x)

    for i in range(m):
        index = vector_jb[i] - 1
        vector_x_new[index] = vector_x[index] - theta * vector_z[i]

    vector_x_new[j0-1] = theta

    return vector_x_new


def main_stage_simplex_method():
    m, n = list(map(int, input().split()))
    all_j = {i for i in range(1, n+1)}

    matrix_a = input_matrix(m)
    vector_b = input_vector(v_b)
    vector_c = input_vector(v_c)
    vector_x = input_float_vector(v_x)
    vector_jb = input_vector(v_jb)

    while True:
        matrix_ab = create_matrix_ab(matrix_a.copy(), vector_jb.copy())

        matrix_b = ban_linalg_inv(matrix_ab.copy())

        vector_cb = create_vector_cb(vector_c.copy(), vector_jb.copy())

        vector_u = vector_cb.dot(matrix_b.copy())

        vector_delta = vector_u.dot(matrix_a.copy()) - vector_c

        j0, min_delta = find_min_delta(vector_delta, all_j, vector_jb)

        if min_delta >= 0:
            return f'Bounded\n{" ".join(map(str, vector_x))}\n'

        j0 = vector_delta.argmin() + 1

        vector_z = matrix_b.dot(matrix_a[:, j0-1])

        theta, s = min_theta(
            vector_x.copy(), vector_z.copy(), vector_jb.copy())

        if theta == -1:
            return 'Unbounded'

        js = vector_jb[s-1]

        vector_x = create_vector_x_new(
            vector_x.copy(), vector_z.copy(), vector_jb.copy(),
            j0, theta, m)

        vector_jb[vector_jb.index(js)] = j0


if __name__ == '__main__':
    t = time.time()
    print(main_stage_simplex_method())
    print(time.time() - t)
