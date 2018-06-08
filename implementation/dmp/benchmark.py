import numpy as np
import matplotlib.pyplot as plt
import dmp




def test_learn_from_demo():
    T = np.linspace(0, 2, 201)
    n_features = 20
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    dmp.initializeRBF(widths, centers, T[-1], T[0], 0.8, 25 / 3.0)

    Y = np.hstack((T[:, np.newaxis], np.hstack((np.sqrt(np.pi * T[:, np.newaxis]), np.sin(np.pi * T[:, np.newaxis])))))
    Y = np.hstack((Y, np.sin(np.pi * T[:, np.newaxis]) * np.sqrt(np.pi * T[:, np.newaxis])))
    weights = np.zeros((n_features, 4))
    alpha = 25.0

    dmp.LearnfromDemo(T, Y, weights, widths, centers, 1e-10, alpha, alpha / 4.0, alpha / 3.0, False)

    last_t = T[0]
    last_y = Y[0].copy()
    last_yd = np.zeros(4)
    last_ydd = np.zeros(4)

    y0 = Y[0].copy()
    y0d = np.zeros(4)
    y0dd = np.zeros(4)

    y = np.empty(4)
    yd = np.empty(4)
    ydd = np.empty(4)

    g = Y[-1].copy()
    gd = np.zeros(4)
    gdd = np.zeros(4)

    Y_replay = []

    for t in np.linspace(T[0], T[-1], T.shape[0]):
        dmp.dmpPropagate(last_t, t,
                         last_y, last_yd, last_ydd,
                         y, yd, ydd,
                         g, gd, gdd,
                         y0, y0d, y0dd,
                         T[-1], T[0],
                         weights,
                         widths,
                         centers,
                         alpha, alpha / 4.0, alpha / 3.0,
                         0.001)

        last_t = t
        last_y[:] = y
        last_yd[:] = yd
        last_ydd[:] = ydd

        Y_replay.append(y.copy())

    Y_replay = np.array(Y_replay)

    distances = np.array([np.linalg.norm(y - y_replay)
                          for y, y_replay in zip(Y, Y_replay)])
    # assert_less(distances.max(), 0.032)
    # assert_less(distances.min(), 1e-10)
    # assert_less(sorted(distances)[len(distances) // 2], 0.02)
    # assert_less(np.mean(distances), 0.02)

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(4, 2)
    axarr[0, 0].plot(T, Y_replay[:, 0])
    axarr[0, 1].plot(T, Y[:, 0])
    axarr[1, 0].plot(T, Y_replay[:, 1])
    axarr[1, 1].plot(T, Y[:, 1])
    axarr[2, 0].plot(T, Y_replay[:, 2])
    axarr[2, 1].plot(T, Y[:, 2])
    axarr[3, 0].plot(T, Y_replay[:, 3])
    axarr[3, 1].plot(T, Y[:, 3])
    plt.show()

test_learn_from_demo()