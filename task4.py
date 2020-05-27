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


def main_stage_simplex_method(matrix_a, vector_b, vector_c, vector_x, vector_jb):
    m = len(matrix_a)
    n = len(matrix_a[0])
    if m == n:
        return vector_x, vector_jb, None
    matrix_ab = matrix_a[:, vector_jb]
    matrix_b = np.linalg.inv(matrix_ab)

    while True:
        vector_jb_n = [i for i in range(n) if i not in vector_jb]

        vector_u = vector_c[vector_jb].dot(matrix_b)
        delta = vector_u.dot(matrix_a[:, vector_jb_n]) - vector_c[vector_jb_n]

        checker = -1
        for i, el in enumerate(delta):
            if el < 0:
                checker = i
                break
        if checker == -1:
            return vector_x, vector_jb, matrix_b

        j0 = vector_jb_n[checker]

        vector_z = matrix_b.dot(matrix_a[:, j0])
        if all([i <= 0 for i in vector_z]):
            return None, None, None

        theta = [vector_x[vector_jb[i]] / vector_z[i] if vector_z[i] > 0 else math.inf for i in range(m)]

        theta_0 = min(theta)
        s = theta.index(theta_0)
        vector_jb[s] = j0

        matrix_b = get_inverted(matrix_b, matrix_a[:, j0], s)

        if matrix_b is None:
            return None, None, None

        vector_x_new = np.zeros(n, dtype=float)
        vector_x_new[vector_jb] = vector_x[vector_jb] - theta_0 * vector_z
        vector_x_new[j0] = theta_0
        vector_x = vector_x_new


def first_and_second_step_simplex_method(matrix_a, vector_b, vector_c, m, n):
    for i in range(m):
        if vector_b[i] < 0:
            vector_b[i] *= -1
            matrix_a[i] *= -1

    vector_jb = [i for i in range(n, n + m)]

    matrix = np.concatenate((matrix_a, np.eye(m)), axis=1)

    vector_c_1 = np.concatenate((np.zeros(n), np.ones(m)))
    vector_x_start = np.concatenate((np.zeros(n), np.copy(vector_b)))

    vector_x, vector_jb, matrix_b = main_stage_simplex_method(
        matrix, vector_b, -vector_c_1, vector_x_start, vector_jb)

    if vector_x is None:
        return None

    vector_x_0 = vector_x[:n]
    vector_x_u = vector_x[n:]

    if any(vector_x_0 < 0) or any(vector_x_u != 0):
        return ()

    while True:
        vector_jb_n = [i for i in range(len(matrix_a[0])) if i not in vector_jb]
        checker = -1
        for i, el in enumerate(vector_jb):
            if el >= n:
                checker = i
                break

        if checker == -1:
            vector_x_new, vector_jb, matrix_b = main_stage_simplex_method(
                matrix_a, vector_b, vector_c, vector_x_0, vector_jb)
            return vector_x_new

        jk = vector_jb[checker]
        i0 = jk - n

        e = np.zeros(len(matrix_a))
        e[i0] = 1
        alphas = e.dot(matrix_b).dot(matrix_a[:, vector_jb_n])

        if all(alphas == 0):
            matrix_a = np.delete(matrix_a, i0, 0)
            matrix_b = np.delete(matrix_b, i0, 1)
            matrix_b = np.delete(matrix_b, checker, 0)
            vector_jb.remove(jk)

            for i, el in enumerate(vector_jb):
                if el > jk:
                    el -= 1
        else:
            i = 0
            for alpha in alphas:
                if alpha != 0:
                    break
                i += 1

            j0 = vector_jb_n[i]
            vector_jb[checker] = j0


def simplex():
    m, n = map(int, input().split())
    matrix_a = input_matrix(m)
    vector_b, vector_c = input_vector(), input_vector()
    vector_x = first_and_second_step_simplex_method(matrix_a, vector_b, np.array(vector_c), m, n)

    if vector_x is None:
        return "Unbounded"
    elif len(vector_x) == 0:
        return "No solution"
    else:
        return f"Bounded\n{' '.join(map(str, vector_x))}"


if __name__ == "__main__":
    print(simplex())
