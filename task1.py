import numpy as np


def input_matrix(n):
    matrix = []
    for el in range(n):
        matrix.append(list(map(float, input().split())))
    return matrix


def input_vector(n):
    return list(map(float, input().split()))


def mofidy_output_matrix(matrix):
    answer = []
    for el in matrix:
        answer.append(' '.join(map(str, el)))
    return '\n'.join(answer)


def sherman_morrison():
    n, i = list(map(int, input().split()))

    matrix_a = np.array(input_matrix(n))
    matrix_b = np.array(input_matrix(n))
    vector_x = np.array(input_vector(n))
    vector_z = matrix_b.dot(vector_x)

    if vector_z[i - 1] == 0:
        return 'NO'

    vector_l = vector_z.copy()
    vector_l[i - 1] = -1

    vector_l_cover = -1 / vector_z[i - 1] * vector_l

    matrix_m = np.eye(n)
    matrix_m[:, i - 1] = vector_l_cover

    return f'YES\n{mofidy_output_matrix(matrix_m.dot(matrix_b))}'


if __name__ == '__main__':
    print(sherman_morrison())
