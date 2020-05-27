import numpy as np
import math


def input_matrix(m):
    matrix = []
    for el in range(m):
        matrix.append(list(map(int, input().split())))
    return np.array(matrix)


def input_vector():
    return np.array(list(map(int, input().split())))


def get_inverted_matrix(matrix_b, vector_x, position):
    vector_l = matrix_b.dot(vector_x.T)

    if vector_l[position] == 0:
        return None

    vector_l_cover = vector_l[position]
    vector_l[position] = -1
    vector_l *= -1 / vector_l_cover
    matrix_b_new = matrix_b + vector_l.reshape(-1, 1) * matrix_b[position]
    matrix_b_new[position] -= matrix_b[position]

    return matrix_b_new


def main_stage_simplex_method(matrix_a, vector_c, vector_x, vector_jb):
    m = len(matrix_a)
    n = len(matrix_a[0])

    if m == n:
        return vector_x, vector_jb, None

    matrix_b = np.linalg.inv(matrix_a[:, vector_jb])
    while True:
        delta = vector_c[vector_jb].dot(matrix_b).dot(matrix_a) - vector_c
        delta[vector_jb] = math.inf

        checker = -1
        for i, el in enumerate(delta):
            if el < 0:
                checker = i
                break
        if checker == -1:
            return vector_x, vector_jb, matrix_b

        j0 = checker

        vector_z = matrix_b.dot(matrix_a[:, j0])

        if all([i <= 0 for i in vector_z]):
            return None, None, None

        theta = [vector_x[vector_jb[i]] / vector_z[i] if vector_z[i] > 0 else math.inf for i in range(m)]
        theta0 = min(theta)

        s = theta.index(theta0)
        vector_jb[s] = j0

        matrix_b = get_inverted_matrix(matrix_b, matrix_a[:, j0], s)

        if matrix_b is None:
            return None, None, None

        vector_x_new = np.zeros(n, dtype=float)
        vector_x_new[vector_jb] = vector_x[vector_jb] - theta0 * vector_z
        vector_x_new[j0] = theta0
        vector_x = vector_x_new


def first_and_second_step_simplex_method(matrix_a, vector_b, vector_c):
    m = len(matrix_a)
    n = len(matrix_a[0])

    for i in range(m):
        if vector_b[i] < 0:
            vector_b[i] *= -1
            matrix_a[i] *= -1

    vector_jb_cover = [i for i in range(n, n + m)]

    vector_c_cover = np.concatenate((np.zeros(n), np.ones(m)))
    vector_x = np.concatenate((np.zeros(n), np.copy(vector_b)))

    vector_x, vector_jb, matrix_b = main_stage_simplex_method(
        np.concatenate((matrix_a, np.eye(m)), axis=1), -vector_c_cover, vector_x, vector_jb_cover)

    if vector_x is None:
        return None

    vector_x_0 = vector_x[:n] 

    if any(vector_x_0 < 0) or any(vector_x[n:] != 0):
        return ()

    while True:
        vector_jb_n = [i for i in range(len(matrix_a[0])) if i not in vector_jb]

        checker = -1
        for i, el in enumerate(vector_jb):
            if el >= n:
                checker = i
                break

        if checker == -1:
            vector_x, vector_jb, matrix_b,  = main_stage_simplex_method(
                matrix_a, vector_c, vector_x_0, vector_jb)
            return vector_x

        el_j_k = vector_jb[checker]
        i0 = el_j_k - n

        vector_e = np.zeros(m)
        vector_e[i0] = 1

        vector_of_alphas = vector_e.dot(matrix_b).dot(matrix_a[:, vector_jb_n])

        if all(vector_of_alphas == 0):
            matrix_a = np.delete(matrix_a, i0, 0)
            matrix_b = np.delete(matrix_b, i0, 1)
            matrix_b = np.delete(matrix_b, checker, 0)
            vector_jb.remove(el_j_k)

            for i in range(len(vector_jb)):
                if vector_jb[i] > el_j_k:
                    vector_jb[i] -= 1
        else:
            i = 0
            for a in vector_of_alphas:
                if a != 0:
                    break
                i += 1

            j0 = vector_jb_n[i]
            vector_jb[checker] = j0


def simplex():
    m, n = map(int, input().split())
    matrix_a = input_matrix(m)
    vector_b = input_vector()
    vector_c = input_vector()

    vector_x = first_and_second_step_simplex_method(matrix_a, vector_b, vector_c)

    if vector_x is None:
        return "Unbounded"
    elif len(vector_x) == 0:
        return "No solution"
    else:
        return f"Bounded\n{' '.join(map(str, vector_x))}"


if __name__ == "__main__":
    print(simplex())
