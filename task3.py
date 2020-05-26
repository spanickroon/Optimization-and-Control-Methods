import numpy as np
import math


def input_matrix(m):
    matrix = []
    for el in range(m):
        matrix.append(list(map(int, input().split())))
    return np.array(matrix)


def input_vector():
    return list(map(int, input().split()))


def get_inverted(matrix_b, vector_x, position):
    vector_l = matrix_b.dot(vector_x.T)
    if vector_l[position] == 0:
        return None

    vector_l_cover = vector_l[position]
    vector_l[position] = -1
    vector_l *= -1 / vector_l_cover

    matrix_b_new = np.eye(len(matrix_b), dtype=float)
    matrix_b_new[:, position] = vector_l

    return matrix_b_new.dot(matrix_b)


def main_stage_simplex_method(m, n, matrix_a, vector_b, vector_c, vector_x, vector_jb):
    if m == n:
        return vector_x
    matrix_ab = matrix_a[:, vector_jb]
    matrix_b = np.linalg.inv(matrix_ab)

    while True:
        vector_jb_n = [i for i in range(n) if i not in vector_jb]

        delta = vector_c[vector_jb].dot(matrix_b).dot(matrix_a[:, vector_jb_n]) - vector_c[vector_jb_n]

        checker = -1
        for i, el in enumerate(delta):
            if el < 0:
                checker = i
                break
        if checker == -1:
            return vector_x

        j0 = vector_jb_n[checker]

        vector_z = matrix_b.dot(matrix_a[:, j0])
        if all([i <= 0 for i in vector_z]):
            return None

        theta = [vector_x[vector_jb[i]] / vector_z[i] if vector_z[i] > 0 else math.inf for i in range(m)]

        theta_0 = min(theta)
        s = theta.index(theta_0)
        vector_jb[s] = j0

        matrix_b = get_inverted(matrix_b, matrix_a[:, j0], s)

        if matrix_b is None:
            return None

        vector_x_new = np.zeros(n, dtype=float)
        vector_x_new[vector_jb] = vector_x[vector_jb] - theta_0 * vector_z
        vector_x_new[j0] = theta_0
        vector_x = vector_x_new


def first_step_simplex_method(matrix_a, vector_b, m, n):
    for i in range(m):
        if vector_b[i] < 0:
            vector_b[i] *= -1
            matrix_a[i] *= -1

    vector_jb = [i for i in range(n, n + m)]
    zeros = [0. for i in range(n)]
    ones = [1. for i in range(m)]

    matrix = np.concatenate((matrix_a, np.eye(m)), axis=1)

    vector_c = np.array(zeros+ones)
    vector_x_start = np.array(zeros+vector_b.copy())

    vector_x = main_stage_simplex_method(
        m, n + m, matrix, vector_b, -vector_c, vector_x_start, vector_jb)

    if vector_x is None:
        return None

    vector_x_0 = vector_x[:n]
    vector_x_u = vector_x[n:]

    if any(vector_x_0 < 0) or any(vector_x_u != 0):
        return ()

    return vector_x_0


def simplex():
    m, n = map(int, input().split())
    matrix_a = input_matrix(m)
    vector_b, vector_c = input_vector(), input_vector()
    vector_x = first_step_simplex_method(matrix_a, vector_b, m, n)

    if vector_x is None:
        return "Unbounded"
    elif len(vector_x) == 0:
        return "No solution"
    else:
        return f"Bounded\n{' '.join(map(str, vector_x))}"


if __name__ == "__main__":
    print(simplex())
