import numpy as np


def exp1(n):
    n = 4
    B_sum = np.array([1, 1, 1, 1])
    R = np.array([[1, -1, 0, 0],
                  [1, 0, -1, 0],
                  [1, 0, 0, -1]])
    S = R @ R.T * 4
    S_inv = np.linalg.inv(S)
    p_mat = R.T @ S_inv @ R
    print("diagonal under unbounded dp", np.diag(p_mat+1.0))

    diff_R = np.array([[0, 0, 0, 0, 0, 0],
                       [2, 1, 1, -1, -1, 0],
                       [1, 2, 1, 1, 0, -1],
                       [1, 1, 2, 0, 1, 1]])
    cov = np.block([[np.ones([1, 1]), np.zeros([1, n-1])],
                    [np.zeros([n-1, 1]), S]])
    cov_inv = np.linalg.inv(cov)
    diff_p_mat = diff_R.T @ cov_inv @ diff_R
    print("diagonal under bounded dp", np.diag(diff_p_mat))
    pass


def exp2():
    B_sum = np.ones([1, 6])
    r1 = np.array([[1, -1, 0],
                   [1, 0, -1]])
    r2 = np.array([[1, -1]])
    B_r1 = np.kron(r1, np.ones([1, 2]))
    S_r1 = B_r1 @ B_r1.T
    B_r2 = np.kron(np.ones([1, 3]), r2)
    S_r2 = B_r2 @ B_r2.T
    print("diagonal under unbounded dp", 1 + 2/3*1/2 + 1/3*1/2)
    B = np.concatenate([B_sum, B_r1, B_r2])

    vec = diff_vec(6)
    diff_R = B @ vec
    diff_R = diff_R.T

    S = np.zeros([4, 4])
    S[0, 0] = 1
    S[1:3, 1:3] = S_r1
    S[3, 3] = S_r2
    S_inv = np.linalg.inv(S)
    diff_p_mat = diff_R @ S_inv @ diff_R.T
    print("diagonal under bounded dp", np.diag(diff_p_mat))


def diff_vec(n):
    size = n*(n-1)//2
    vec = np.zeros([n, size])
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            vec[i, k] = 1
            vec[j, k] = -1
            k += 1
    return vec


def subtract_matrix(k):
    """Return Subtraction matrix Sub_k."""
    mat = np.zeros([k-1, k])
    for i in range(k-1):
        mat[i, i] = 1
        mat[i, i+1] = -1
    return mat


if __name__ == '__main__':
    n = 4
    # B_sum = np.array([[1, 1, 1, 1]])
    # R = np.array([[1, -1, 0, 0],
    #               [1, 0, -1, 0],
    #               [1, 0, 0, -1]])
    B_sum = np.ones([1, n])
    R = subtract_matrix(n)
    S = R @ R.T
    S_inv = np.linalg.inv(S)
    p_mat = R.T @ S_inv @ R
    print("diagonal under unbounded dp", np.diag(p_mat + 1.0/n**2))

    B = np.concatenate([B_sum, R])
    diff_R = B @ diff_vec(n)
    # diff_R = np.array([[0, 0, 0, 0, 0, 0],
    #                    [2, 1, 1, -1, -1, 0],
    #                    [1, 2, 1, 1, 0, -1],
    #                    [1, 1, 2, 0, 1, 1]])
    cov = np.block([[np.ones([1, 1])*n**2, np.zeros([1, n - 1])],
                    [np.zeros([n - 1, 1]), S]])
    cov_inv = np.linalg.inv(cov)
    diff_p_mat = diff_R.T @ cov_inv @ diff_R
    print("diagonal under bounded dp", np.diag(diff_p_mat))
