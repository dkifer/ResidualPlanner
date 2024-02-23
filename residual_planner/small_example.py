import numpy as np


if __name__ == '__main__':
    Q1 = np.array([[1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1]])
    Q2 = np.array([[1, 0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0, 1]])

    R0 = np.array([[1, 1, 1, 1, 1, 1]])
    R1 = np.array([[1, 1, 1, -1, -1, -1]])
    R2 = np.array([[1, -1, 0, 1, -1, 0],
                   [1, 0, -1, 1, 0, -1]])
    x = np.array([200, 300, 100, 220, 260, 80])

    w0 = R0 @ x + np.random.normal(0, 1.52, [1])
    w1 = R1 @ x + np.random.normal(0, 1.96, [1])
    mean = np.array([0, 0])
    cov = np.array([[2, 1], [1, 2]]) * 0.8
    w2 = R2 @ x + np.random.multivariate_normal(mean, cov)

    U10 = Q1 @ np.linalg.pinv(R0)
    U11 = Q1 @ np.linalg.pinv(R1)
    U20 = Q2 @ np.linalg.pinv(R0)
    U22 = Q2 @ np.linalg.pinv(R2)

    y1 = U10 @ w0 + U11 @ w1
    y2 = U20 @ w0 + U22 @ w2
