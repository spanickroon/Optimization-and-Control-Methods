import numpy as np

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


def input_matrix(m):
    matrix = []
    for el in m_a:
        matrix.append(list(map(int, el.split())))
    return matrix


def input_vector(v):
    return list(map(int, v.split()))


def input_float_vector(v):
    return list(map(float, v.split()))


def create_matrix_ab(matrix_a, vector_jb):
    size = len(vector_jb)
    matrix_ab = np.zeros((size, size))

    for i, el in enumerate(vector_jb):
        matrix_ab[:, i] = matrix_a[:, el-1]

    return matrix_ab


def create_vector_cb(vector_c, vector_jb):
    vector_cb = np.zeros(len(vector_jb))

    for i, el in enumerate(vector_jb):
        vector_cb[i] = vector_c[el-1]

    return vector_cb


def create_vector_delta(vector_u, matrix_a, vector_c):
    vector_ua = vector_u.dot(matrix_a)
    vector_delta = np.zeros(len(vector_ua))

    for i, el in enumerate(vector_ua):
        vector_delta[i] = vector_ua[i] - vector_c[i]

    return vector_delta


def main_stage_simplex_method():
    m, n = list(map(int, input().split()))
    all_j = {i for i in range(1, n+1)}

    matrix_a = np.array(input_matrix(m))
    vector_b = np.array(input_vector(v_b))
    vector_c = np.array(input_vector(v_c))
    vector_x = np.array(input_float_vector(v_x))
    vector_jb = np.array(input_vector(v_jb))

    while True:
        matrix_ab = create_matrix_ab(matrix_a.copy(), vector_jb.copy())
        matrix_b = np.linalg.inv(matrix_ab.copy())
        vector_cb = create_vector_cb(vector_c.copy(), vector_jb.copy())

        vector_u = vector_cb.dot(matrix_b)
        vector_delta = create_vector_delta(
            vector_u.copy(),
            matrix_a.copy(),
            vector_c.copy()
            )

        negative_delta = []

        for i, el in enumerate(vector_delta):
            if el < 0:
                negative_delta.append(i+1)

        if len(negative_delta) == 0:
            return f'Bound\n{" ".join(map(str, vector_x))}'

        vector_j_h = np.array(list(all_j - set(vector_jb)))

        j0 = set(negative_delta).intersection(set(vector_j_h)).pop()

        vector_z = matrix_b.dot(matrix_a[:, j0-1])

        if len(list(filter(lambda x: x > 0, vector_z))) == 0:
            return 'Unbound'

        vector_theta = []
        s = 1

        for i, el in enumerate(vector_jb):
            if vector_z[i] > 0:
                vector_theta.append(vector_x[el-1] / vector_z[i])
                if vector_theta[-1] <= min(vector_theta):
                    s = i + 1

        js = {vector_jb[s-1]}
        theta = min(vector_theta)

        vector_x_new = []

        i = 0
        for j, el in enumerate(vector_x):
            if j + 1 == j0:
                vector_x_new.append(theta)
            elif j + 1 in set(vector_j_h) - {j0}:
                vector_x_new.append(0)
            else:
                vector_x_new.append(vector_x[j] - vector_z[i] * theta)
                i += 1

        vector_jb_new = np.array(sorted((set(vector_jb) - js).union({j0})))

        vector_jb = vector_jb_new.copy()
        vector_x = vector_x_new.copy()

        print(matrix_a)
        print(matrix_ab)
        print(matrix_b)
        print(vector_u)
        print(vector_delta)
        print(vector_j_h)
        print(vector_z)
        print(j0)
        print(js)
        print(vector_jb_new)
        print(vector_x)


if __name__ == '__main__':
    print(main_stage_simplex_method())
