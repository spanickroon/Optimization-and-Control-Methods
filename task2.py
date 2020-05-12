import numpy as np


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
    for el in range(m):
        matrix.append(list(map(int, input().split())))
    return np.array(matrix)


def input_vector():
    return list(map(int, input().split()))


def input_float_vector():
    return list(map(float, input().split()))


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
    vector_b = input_vector()
    vector_c = input_vector()
    vector_x = input_float_vector()
    vector_jb = input_vector()

    while True:
        matrix_ab = create_matrix_ab(matrix_a.copy(), vector_jb.copy())
        try:
            matrix_b = np.linalg.inv(matrix_ab.copy())
        except np.linalg.LinAlgError:
            print('Unbounded')
            exit()

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
    print(main_stage_simplex_method())
