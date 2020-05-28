import numpy as np
import math
from operator import sub


def input_matrix(size):
    matrix = []
    for el in range(size):
        matrix.append(list(map(float, input().split())))
    return np.array(matrix)


def input_vector():
    return list(map(int, input().split()))


def iterative_algorithm(
        m, n, matrix_a, matrix_d, vector_b,
        vector_c, vector_x, vector_j, vector_j_opt):

    eps = 1 * 10 ** -5

    while True:
        vector_c_x = vector_c + matrix_d.dot(vector_x)
        B = np.linalg.inv(matrix_a[:, vector_j])

        delta = -vector_c_x[vector_j].dot(B).dot(matrix_a) + vector_c_x

        j0 = -1

        for i in range(n):
            if delta[i] + eps < 0:
                j0 = i
                break
        if j0 == -1:
            return vector_x

        vector_l = np.zeros(n)
        vector_l[j0] = 1

        matrix_a_opt = np.array(matrix_a[:, vector_j_opt])
        matrix_h = np.vstack((
            np.hstack((matrix_d[vector_j_opt][:, vector_j_opt], matrix_a_opt.T)),
            np.hstack((matrix_a_opt, np.zeros((m, m))))
            ))

        vector_b = np.append(matrix_d[vector_j_opt, j0], matrix_a[:, j0])
        vector_x_new = - np.linalg.inv(matrix_h).dot(vector_b)

        for i in range(len(vector_j_opt)):
            vector_l[vector_j_opt[i]] = vector_x_new[i]

        vector_d = vector_l.transpose().dot(matrix_d).dot(vector_l)

        thetas = [math.inf for i in range(n)]
        for i in vector_j_opt:
            thetas[i] = -vector_x[i]/vector_l[i] if vector_l[i] < -eps else math.inf

        thetas[j0] = math.inf if abs(vector_d) < eps else abs(delta[j0]) / vector_d

        min_theta = min(thetas)
        checker = thetas.index(min_theta)

        if min_theta == math.inf:
            return None

        vector_x = vector_x + min_theta * vector_l

        if checker == j0:
            vector_j_opt.append(checker)
        elif checker in vector_j_opt and checker not in vector_j:
            vector_j_opt.remove(checker)
        else:
            s = vector_j.index(checker)
            value_j_index = -1

            for j in [j for j in vector_j_opt if j not in vector_j]:
                if abs((B.dot(matrix_a[:, j]))[s]) > eps:
                    value_j_index = j
                    break

            if value_j_index != -1:
                vector_j[s] = value_j_index
                vector_j_opt.remove(checker)
            else:
                vector_j_opt[vector_j_opt.index(checker)] = j0
                vector_j[s] = j0


def quadratic_programming():
    m, n, k = map(int, input().split())

    matrix_a = input_matrix(m)
    vector_b = np.array(input_vector())
    vector_c = np.array(input_vector())
    matrix_d = input_matrix(n)
    vector_x = np.array(list(map(float, input().split())))
    vector_j = list(map(sub, input_vector(), [1 for i in range(m)]))
    vector_j_opt = list(map(sub, input_vector(), [1 for i in range(m)]))

    vector_x_new = iterative_algorithm(
        m, n, matrix_a, matrix_d,
        vector_b, vector_c, vector_x, vector_j, vector_j_opt)

    if vector_x_new is None:
        return "Unbounded"
    else:
        return f"Bounded\n{' '.join(map(str, vector_x_new))}"


if __name__ == "__main__":
    print(quadratic_programming())
