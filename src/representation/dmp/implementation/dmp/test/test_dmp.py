import numpy as np
import dmp
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises_regexp, assert_less, assert_almost_equal
import matplotlib.pyplot as plt


def test_compute_gradient():

    n_steps = 101

    T = np.linspace(0, 1, n_steps)
    input = np.zeros((n_steps, 2))
    input[:, 0] = np.cos(2 + np.pi * T)
    input[:, 1] = np.sin(2 + np.pi * T)

    g = np.zeros_like(input)
    dmp.compute_gradient(input, g, T, True)


    g_hat = np.vstack((np.zeros(2), np.diff(input, axis=0) / np.diff(T)[:, np.newaxis]))
    assert_array_almost_equal(g, g_hat)

    g_int = np.vstack((input[0], np.cumsum(g[1:] * np.diff(T)[:, np.newaxis], axis=0) + input[0]))
    assert_array_almost_equal(input, g_int)


def test_compute_quaterion_gradient():

    n_steps = 101

    T = np.linspace(0, 1, n_steps)
    R = np.zeros((n_steps, 4))


    for i in range (0, n_steps):
        angle = T[i] * np.pi
        R[i, 0] = np.cos(angle / 2)
        R[i, 1] = np.sqrt(2) / 2.0 * np.sin(angle / 2)
        R[i, 2] = 1 / 2.0 * np.sin(angle / 2)
        R[i, 3] = 1 / 2.0 * np.sin(angle / 2)


    R_g = np.zeros((n_steps, 3))
    dmp.compute_quaternion_gradient(R, R_g, T, True)

    R_g_integral = np.zeros_like(R)
    R_g_integral[0] = R[0]


    for i in range (1, n_steps):
        delta_w = R_g[i, :] * (T[i] - T[i - 1]) / 2
        n = np.linalg.norm(delta_w)
        v = np.sin(n) * delta_w / n

        R_g_integral[i, 0] = np.cos(n) * R_g_integral[i - 1, 0] - np.dot(v, R_g_integral[i - 1, 1:])
        R_g_integral[i, 1:] = np.cos(n) * R_g_integral[i - 1, 1:] + v * R_g_integral[i - 1, 0] + np.cross(v, R_g_integral[i - 1, 1:])
    assert_array_almost_equal(R, R_g_integral)

def test_compute_invalid_final_phase():
    assert_raises_regexp(ValueError, "Final phase must be > 0!", dmp.computeAlphaX, 0.0, 0.0, 0.446)

def test_comput_alpha_x():
    alpha = dmp.computeAlphaX(0.01, 0.446, 0.0)
    assert_almost_equal(4.5814764780043, alpha)

def test_initialize_rbf_too_few_weights():
    n_features = 1
    widths = np.zeros(n_features)
    centers = np.zeros(n_features)
    assert_raises_regexp(ValueError, "...", dmp.initializeRBF, widths, centers, 0.1, 0, 0.6, 25.0 / 3.0)

def test_initialize_rbf_invalid_times():
    n_features = 100
    widths = np.zeros(n_features)
    centers = np.zeros(n_features)
    assert_raises_regexp(ValueError, "...", dmp.initializeRBF, widths, centers, 0, 0.1, 0.6,  25.0 / 3.0)

def test_initialize_rbf():
    widths = np.empty(10)
    centers = np.empty(10)
    dmp.initializeRBF(widths, centers, 0.446, 0.0, 0.8, 4.5814764780043)
    assert_array_almost_equal(
        widths,
        np.array([
            1.39105769528669, 3.87070066903258, 10.7704545397461,
            29.969429545615, 83.3917179609352, 232.042408878407,
            645.671786535403, 1796.62010036397, 4999.20215246208,
            4999.20215246208])
    )
    assert_array_almost_equal(
        centers,
        np.array([ 1.        ,  0.59948425,  0.35938137,  0.21544347,  0.12915497,
        0.07742637,  0.04641589,  0.02782559,  0.01668101,  0.01      ])
    )


def test_learn_ill_conditioning():
    n_features = 101
    widths = np.zeros(n_features)
    centers = np.zeros(n_features)
    dmp.initializeRBF(widths, centers, 1.0, 0.0, 0.8, 25 / 3.0)

    T = np.linspace(0, 1, 101)
    Y = np.hstack((T[:, np.newaxis], np.cos(2 * T[:, np.newaxis])))

    weights = np.empty((n_features, 2))

    alpha = 25

    assert_raises_regexp(
        ValueError, "must be >= 0",
        dmp.LearnfromDemo, T, Y, weights, widths, centers, -1.0, alpha,
        alpha / 4.0, alpha / 3.0, False)
    assert_raises_regexp(
        ValueError, "instable",
        dmp.LearnfromDemo, T, Y, weights, widths, centers, 0.0, alpha,
        alpha / 4.0, alpha / 3.0, False)


def test_propagate_invalid_times():
    last_y = np.array([0.0])
    last_yd = np.array([0.0])
    last_ydd = np.array([0.0])

    y = np.empty(1)
    yd = np.empty(1)
    ydd = np.empty(1)

    g = np.array([1.0])
    gd = np.array([1.0])
    gdd = np.array([1.0])

    n_weights = 10
    weights = np.zeros((n_weights, 1))
    widths = np.zeros(n_weights)
    centers = np.zeros(n_weights)
    alpha = 25

    dmp.initializeRBF(widths, centers, 1.0 , 0.0, 0.8, alpha)

    assert_raises_regexp(
        ValueError, "Goal must be chronologically after start", dmp.dmpPropagate,
        0.0, 0.1,
        last_y, last_yd, last_ydd,
        y, yd, ydd,
        g, gd, gdd,
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
        0.0, 0.5,
        weights,
        widths,
        centers,
        alpha, alpha / 4.0, alpha / 3.0,
        0.001
    )


def test_propagate():
    last_y = np.array([0.0])
    last_yd = np.array([0.0])
    last_ydd = np.array([0.0])

    y = np.empty(1)
    yd = np.empty(1)
    ydd = np.empty(1)

    g = np.array([1.0])
    gd = np.array([0.0])
    gdd = np.array([0.0])

    n_weights = 10
    weights = np.zeros((n_weights, 1))
    widths = np.zeros(n_weights)
    centers = np.zeros(n_weights)
    alpha = 25

    execution_time = 1.0

    dmp.initializeRBF(widths, centers, execution_time, 0.0, 0.8, alpha / 3.0)


    last_t = 0.0
    dt = 0.001
    for t in np.linspace(0, 1.5 * execution_time, 1000):
        dmp.dmpPropagate(last_t , t,
        last_y, last_yd, last_ydd,
        y, yd, ydd,
        g, gd, gdd,
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
        execution_time, 0.0,
        weights,
        widths,
        centers,
        alpha, alpha / 4.0, alpha / 3.0,
        dt)

        last_t = t
        last_y[:] = y
        last_yd[:] = yd
        last_ydd[:] = ydd

    assert_array_almost_equal(y, g, decimal=6)
    assert_array_almost_equal(yd, gd, decimal=5)
    assert_array_almost_equal(ydd, gdd, decimal=4)


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


# def test_quaternion_propagate_invalid_times():
#     last_r = np.array([0.0, 1.0, 0.0, 0.0])
#     last_rd = np.array([0.0, 0.0, 0.0])
#     last_rdd = np.array([0.0, 0.0, 0.0])
#
#     r = np.array([0.0, 1.0, 0.0, 0.0])
#     rd = np.array([0.0, 0.0, 0.0])
#     rdd = np.array([0.0, 0.0, 0.0])
#
#     g = np.array([0.0, 0.0, 7.07106781e-01, 7.07106781e-01])
#     gd = np.array([0.0, 0.0, 0.0])
#     gdd = np.array([0.0, 0.0, 0.0])
#
#     r0 = np.array([0.0, 1.0, 0.0, 0.0])
#     r0d = np.array([0.0, 0.0, 0.0])
#     r0dd = np.array([0.0, 0.0, 0.0])
#
#     n_features = 10
#     weights = np.zeros((n_features, 3))
#     execution_time = 1.0
#     alpha = 25.0
#
#     widths = np.empty(n_features)
#     centers = np.empty(n_features)
#
#     dmp.initializeRBF(widths, centers, execution_time, 0.0, 0.8, alpha / 3.0)
#
#     T = np.linspace(0.0, 1.0 * execution_time, 1001)
#
#     last_t = 0.0
#     R_replay = []
#     for t in T:
#         dmp.dmpPropagateQuaternion(
#             last_t, t,
#             last_r, last_rd, last_rdd,
#             r, rd, rdd,
#             g, gd, gdd,
#             r0, r0d, r0dd,
#             execution_time, 0.0,
#             weights,
#             widths,
#             centers,
#             alpha, alpha / 4.0, alpha / 3.0,
#             0.001
#         )
#         last_t = t
#         last_r[:] = r
#         last_rd[:] = rd
#         last_rdd[:] = rdd
#         R_replay.append(r.copy())
#
#     R_replay = np.asarray(R_replay)
#     f, axarr = plt.subplots(4, 2)
#     axarr[0, 0].plot(T, R_replay[:, 0])
#     axarr[1, 0].plot(T, R_replay[:, 1])
#     axarr[2, 0].plot(T, R_replay[:, 2])
#     axarr[3, 0].plot(T, R_replay[:, 3])
#     assert_array_almost_equal(r, g, decimal=4)
#     assert_array_almost_equal(rd, gd, decimal=3)
#
#
#     assert_array_almost_equal(rdd, gdd, decimal=2)

# last_r = np.array([0.0, 1.0, 0.0, 0.0])
# last_rd = np.array([0.0, 0.0, 0.0])
# last_rdd = np.array([0.0, 0.0, 0.0])
#
# r = np.array([0.0, 1.0, 0.0, 0.0])
# rd = np.array([0.0, 0.0, 0.0])
# rdd = np.array([0.0, 0.0, 0.0])
#
# g = np.array([0.0, 0.0, 7.07106781e-01, 7.07106781e-01])
# gd = np.array([0.0, 0.0, 0.0])
# gdd = np.array([0.0, 0.0, 0.0])
#
# r0 = np.array([0.0, 1.0, 0.0, 0.0])
# r0d = np.array([0.0, 0.0, 0.0])
# r0dd = np.array([0.0, 0.0, 0.0])
#
# n_features = 10
# weights = np.zeros((n_features, 3))
# execution_time = 1.0
# alpha = 25.0
#
# widths = np.empty(n_features)
# centers = np.empty(n_features)
#
# dmp.initializeRBF(widths, centers, execution_time, 0.0, 0.8, alpha / 3.0)
#
# T = np.linspace(0.0, 1.0 * execution_time, 1001)
#
# last_t = 0.0
# R_replay = []
# for t in T:
#     dmp.dmpPropagateQuaternion(
#         last_t, t,
#         last_r, last_rd, last_rdd,
#         r, rd, rdd,
#         g, gd, gdd,
#         r0, r0d, r0dd,
#         execution_time, 0.0,
#         weights,
#         widths,
#         centers,
#         alpha, alpha / 4.0, alpha / 3.0,
#         0.001
#     )
#     last_t = t
#     last_r[:] = r
#     last_rd[:] = rd
#     last_rdd[:] = rdd
#     R_replay.append(r.copy())
#
# R_replay = np.asarray(R_replay)
# f, axarr = plt.subplots(4, 2)
# axarr[0, 0].plot(T, R_replay[:, 0])
# axarr[1, 0].plot(T, R_replay[:, 1])
# axarr[2, 0].plot(T, R_replay[:, 2])
# axarr[3, 0].plot(T, R_replay[:, 3])

# def test_learn_from_demo_quaternion():
#     T = np.linspace(0, 2, 201)
#     n_features = 20
#     widths = np.empty(n_features)
#     centers = np.empty(n_features)
#     dmp.initializeRBF(widths, centers, T[-1], T[0], 0.8, 25 / 3.0)
#     R = np.zeros((T.shape[0], 4))
#     for i in range(0, T.shape[0]):
#                 angle = T[i] * np.pi
#                 R[i] = np.array([np.cos(angle  / 2.01),
#                      np.sqrt(0.5) * np.sin(angle / 2.01),
#                      0.5 * np.sin(angle / 2.01),
#                      0.5 * np.sin(angle / 2.01)])
#
#     weights = np.zeros((n_features, 3))
#     alpha = 25.0
#
#     dmp.LearnfromDemoQuaternion(T, R, weights, widths, centers, 1e-10, alpha, alpha / 4.0, alpha / 3.0, False)
#
#     last_t = T[0]
#     last_y = R[0].copy()
#     last_yd = np.zeros(3)
#     last_ydd = np.zeros(3)
#
#     y0 = R[0].copy()
#     y0d = np.zeros(3)
#     y0dd = np.zeros(3)
#
#     y = np.empty(4)
#     yd = np.empty(3)
#     ydd = np.empty(3)
#
#     g = R[-1].copy()
#     gd = np.zeros(3)
#     gdd = np.zeros(3)
#
#     R_replay = []
#
#     for t in np.linspace(T[0], T[-1], T.shape[0]):
#         dmp.dmpPropagateQuaternion(last_t, t,
#                                last_y, last_yd, last_ydd,
#                                y, yd, ydd,
#                                g, gd, gdd,
#                                y0, y0d, y0dd,
#                                T[-1], T[0],
#                                weights,
#                                widths,
#                                centers,
#                                alpha, alpha / 4.0, alpha / 3.0,
#                                0.001)
#
#         last_t = t
#         last_y[:] = y
#         last_yd[:] = yd
#         last_ydd[:] = ydd
#
#         R_replay.append(y.copy())
#
#     R_replay = np.array(R_replay)
#
#     distances = np.array([np.linalg.norm(y - y_replay)
#                       for y, y_replay in zip(R, R_replay)])
#     # assert_less(distances.max(), 0.032)
#     # assert_less(distances.min(), 1e-10)
#     # assert_less(sorted(distances)[len(distances) // 2], 0.02)
#     # assert_less(np.mean(distances), 0.02)
#
#     # Four axes, returned as a 2-d array
#     f, axarr = plt.subplots(4, 2)
#     axarr[0, 0].plot(T, R_replay[:, 0])
#     axarr[0, 1].plot(T, R[:, 0])
#     axarr[1, 0].plot(T, R_replay[:, 1])
#     axarr[1, 1].plot(T, R[:, 1])
#     axarr[2, 0].plot(T, R_replay[:, 2])
#     axarr[2, 1].plot(T, R[:, 2])
#     axarr[3, 0].plot(T, R_replay[:, 3])
#     axarr[3, 1].plot(T, R[:, 3])
#     plt.show()


def test_imitate_quaternion():
    T = np.linspace(0, 2, 201)
    n_features = 20
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    dmp.initializeRBF(widths, centers, T[-1], T[0], 0.8, 25 / 3.0)
    R = np.zeros((T.shape[0], 4))
    for i in range(0, T.shape[0]):
                angle1 = T[i] * np.pi
                R[i] = np.array([np.cos(angle  / 2.01),
                     np.sqrt(0.5) * np.sin(angle / 2.01),
                     0.5 * np.sin(angle / 2.01),
                     0.5 * np.sin(angle / 2.01)])

    weights = np.zeros((n_features, 3))
    alpha = 25.0

    dmp.LearnfromDemoQuaternion(T, R, weights, widths, centers, 1e-10, alpha, alpha / 4.0, alpha / 3.0, False)

    last_t = T[0]
    last_y = R[0].copy()
    last_yd = np.zeros(3)
    last_ydd = np.zeros(3)

    y0 = R[0].copy()
    y0d = np.zeros(3)
    y0dd = np.zeros(3)

    y = np.empty(4)
    yd = np.empty(3)
    ydd = np.empty(3)

    g = R[-1].copy()
    gd = np.zeros(3)
    gdd = np.zeros(3)

    R_replay = []

    for t in np.linspace(T[0], T[-1], T.shape[0]):
        dmp.dmpPropagateQuaternion(last_t, t,
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

        R_replay.append(y.copy())

    R_replay = np.array(R_replay)

    distances = np.array([np.linalg.norm(y - y_replay)
                      for y, y_replay in zip(R, R_replay)])
    # assert_less(distances.max(), 0.032)
    # assert_less(distances.min(), 1e-10)
    # assert_less(sorted(distances)[len(distances) // 2], 0.02)
    # assert_less(np.mean(distances), 0.02)

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(4, 2)
    axarr[0, 0].plot(T, R_replay[:, 0])
    axarr[0, 1].plot(T, R[:, 0])
    axarr[1, 0].plot(T, R_replay[:, 1])
    axarr[1, 1].plot(T, R[:, 1])
    axarr[2, 0].plot(T, R_replay[:, 2])
    axarr[2, 1].plot(T, R[:, 2])
    axarr[3, 0].plot(T, R_replay[:, 3])
    axarr[3, 1].plot(T, R[:, 3])
    plt.show()






