import os
import numpy as np

##hackey way
import sys
sys.path.append('/home/yzheng/PycharmProjects/DMP/DMP')

from dmp_behavior import DMPBehavior
from nose.tools import assert_equal, assert_raises_regexp
from numpy.testing import assert_array_equal, assert_array_almost_equal
import matplotlib.pyplot as plt

CURRENT_PATH = os.sep.join(__file__.split(os.sep)[:-1])
DMP_CONFIG_FILE = CURRENT_PATH + os.sep + "dmp_model.yaml"
if not CURRENT_PATH:
    DMP_CONFIG_FILE = "dmp_model.yaml"

n_task_dims = 1

def eval_loop(beh, xva):
    beh.set_inputs(xva)
    beh.step()
    beh.get_outputs(xva)

# def test_dmp_dimensions_do_not_match():
#     beh = DMPBehavior()
#     assert_raises_regexp(ValueError, "Input and output dimensions much match, got {} inputs and {} outputs."
#                              .format(1, 2),
#                          beh.init, 1, 2)
#
# # def test_dmp_default():
# #     beh = DMPBehavior()
# #     beh.init(3 * n_task_dims, 3 * n_task_dims)
# #
# #     xva = np.zeros(3 * n_task_dims)
# #     beh.reset()
# #     t = 0
# #     while beh.can_step():
# #         eval_loop(beh, xva)
# #         t += 1
# #
# #     assert_equal(t, 101)
# #     assert_array_equal(xva[:n_task_dims], np.zeros(n_task_dims))
# #     assert_array_equal(xva[n_task_dims:-n_task_dims], np.zeros(n_task_dims))
# #
# #
# #     assert_array_equal(xva[-n_task_dims:], np.zeros(n_task_dims))
#
# def test_dmp_get_set_params():
#     beh = DMPBehavior()
#     beh.init(3 * n_task_dims, 3 * n_task_dims)
#
#     assert_equal(beh.get_n_params(), 50 * n_task_dims)
#     params = beh.get_params()
#     assert_array_equal(params, np.zeros((50, n_task_dims)))
#
#     random_state = np.random.RandomState(0)
#     expected_params = random_state.randn(50 * n_task_dims)
#     beh.set_params(expected_params)
#
#     actual_params = beh.get_params()
#     assert_array_equal(actual_params, expected_params.reshape((50, n_task_dims)))
#
# def test_dmp_from_config():
#     beh = DMPBehavior(yaml_config=DMP_CONFIG_FILE)
#     beh.init(18, 18)
#
#     xva = np.zeros(18)
#     beh.reset()
#     t = 0
#
#     while beh.can_step():
#         eval_loop(beh, xva)
#         t += 1
#
#
#     assert_equal(t, 447)
#
# def test_dmp_constructor_args():
#     beh = DMPBehavior(execution_time=2)
#     beh.init(3 * n_task_dims, 3 * n_task_dims)
#
#     xva = np.zeros(3 * n_task_dims)
#     beh.reset()
#     t = 0
#     while beh.can_step():
#         eval_loop(beh, xva)
#         t += 1
#
#     assert_equal(t, 201)
#
# def test_dmp_metaparameter_not_permitted():
#     beh = DMPBehavior()
#     beh.init(3, 3)
#     assert_raises_regexp(ValueError, "Meta parameter .* is not allowed",
#     beh.set_meta_parameters, ["unknown"], [None])
#
# def test_dmp_change_goal():
#     beh = DMPBehavior()
#     beh.init(3 * n_task_dims, 3 * n_task_dims)
#
#     beh.set_meta_parameters(["g"], [np.ones(n_task_dims) * 1.5])
#
#     xva = np.zeros(3 * n_task_dims)
#     beh.reset()
#     while beh.can_step():
#         eval_loop(beh, xva)
#     for _ in range(30):  # Convergence
#         eval_loop(beh, xva)
#
#     assert_array_almost_equal(xva[:n_task_dims], np.ones(n_task_dims) * 1.5,
#                               decimal=4)
#     assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
#                               np.zeros(n_task_dims), decimal=2)
#     assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
#                               decimal=2)


# def test_dmp_change_goal_velocity():
#     beh = DMPBehavior()
#     beh.init(3 * n_task_dims, 3 * n_task_dims)
#
#     beh.set_meta_parameters(["gd"], [np.ones(n_task_dims)])
#
#     xva = np.zeros(3 * n_task_dims)
#     beh.reset()
#     Y_replay = []
#     Yd_replay = []
#     beh.execution_time = 2
#     while beh.can_step():
#         eval_loop(beh, xva)
#         Y_replay.append(list(xva[:beh.n_task_dims]))
#         Yd_replay.append(list(xva[beh.n_task_dims : -beh.n_task_dims]))
#
#     Y_replay = np.array(Y_replay)
#     Yd_replay = np.array(Yd_replay)
#     f, axarr = plt.subplots(2, 2)
#     axarr[0, 0].plot(np.linspace(0, beh.execution_time + beh.dt, len(Y_replay)),
#                      Y_replay)
#
#     axarr[0, 1].plot(np.linspace(0, beh.execution_time + beh.dt, len(Y_replay)),
#                      Yd_replay)
#     plt.show()
#
#
#
#     assert_array_almost_equal(xva[:n_task_dims], np.zeros(n_task_dims),
#                               decimal=2)
#     # assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
#     #                           np.ones(n_task_dims), decimal=1)
#     assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
#                               decimal=0)


def test_dmp_change_execution_time():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_meta_parameters(["y0"], [np.ones(n_task_dims)])
    X1 = beh.gen_traj()[0]
    beh.set_meta_parameters(["execution_time"], [2.0])
    X2 = beh.gen_traj()[0]

    # plt.plot(np.linspace(0, 2 + 0.01, 201), X2, 'r',
    #          np.linspace(0, 1 + 0.01, 101), X1, 'b')
    # plt.show()

    assert_equal(X2.shape[0], 201)
    # assert_array_almost_equal(X1, X2[::2], decimal=3)
    #assert (None != None)

# def test_dmp_change_weights():
#     beh = DMPBehavior()
#     beh.init(3 * n_task_dims, 3 * n_task_dims)
#
#     beh.set_params(np.ones(50 * n_task_dims))
#
#     xva = np.zeros(3 * n_task_dims)
#     beh.reset()
#     Y_replay = []
#     Yd_replay = []
#     while beh.can_step():
#         eval_loop(beh, xva)
#         Y_replay.append(list(xva[:beh.n_task_dims]))
#         Yd_replay.append(list(xva[beh.n_task_dims : -beh.n_task_dims]))
#
#     Y_replay = np.array(Y_replay)
#     Yd_replay = np.array(Yd_replay)
#     f, axarr = plt.subplots(1, 2)
#     axarr[0, 0].plot(np.linspace(0, beh.execution_time + beh.dt, len(Y_replay)),
#                          Y_replay)
#
#     axarr[0, 1].plot(np.linspace(0, beh.execution_time + beh.dt, len(Y_replay)),
#                          Yd_replay)
#     plt.show()
#
#     assert_array_almost_equal(xva[:n_task_dims], np.zeros(n_task_dims),
#                               decimal=3)
#     assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
#                               np.zeros(n_task_dims), decimal=2)
#     assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
# decimal=1)

# def test_dmp_set_meta_params_before_init():
#     beh = DMPBehavior()
#
#     x0 = np.ones(n_task_dims) * 0.43
#     g = np.ones(n_task_dims) * -0.21
#     gd = np.ones(n_task_dims) * 0.12
#     execution_time = 1.5
#
#     beh.set_meta_parameters(["y0", "g", "gd", "execution_time"],
#                             [x0, g, gd, execution_time])
#     beh.init(3 * n_task_dims, 3 * n_task_dims)
#
#     xva = np.zeros(3 * n_task_dims)
#     xva[:n_task_dims] = x0
#
#     beh.reset()
#     Y_replay = []
#     Yd_replay = []
#     Ydd_replay = []
#     t = 0
#     while beh.can_step():
#         eval_loop(beh, xva)
#         Y_replay.append(list(xva[:beh.n_task_dims]))
#         Yd_replay.append(list(xva[beh.n_task_dims: -beh.n_task_dims]))
#         Ydd_replay.append(list(xva[-beh.n_task_dims:]))
#         t += 1
#
#     # for _ in range (0, 30):
#     #     eval_loop(beh, xva)
#     #     Y_replay.append(list(xva[:beh.n_task_dims]))
#     #     Yd_replay.append(list(xva[beh.n_task_dims: -beh.n_task_dims]))
#     #     Ydd_replay.append(list(xva[-beh.n_task_dims:]))
#     #     t += 1
#
#     Y_replay = np.array(Y_replay)
#     Yd_replay = np.array(Yd_replay)
#     Ydd_replay = np.array(Ydd_replay)
#     print Yd_replay
#     f, axarr = plt.subplots(1, 2)
#     # axarr[0, 0].plot(np.linspace(0, beh.execution_time + beh.dt, len(Y_replay)),
#     #                  Y_replay)
#     #
#     # axarr[0, 1].plot(np.linspace(0, beh.execution_time + beh.dt, len(Y_replay)),
#     #                  Yd_replay)
#     # plt.show()
#     assert_array_almost_equal(xva[:n_task_dims], g, decimal=3)
#     assert_array_almost_equal(xva[n_task_dims:-n_task_dims], gd, decimal=2)
#     assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
#                               decimal=1)
#     assert_equal(t, 151)

# def test_dmp_more_steps_than_allowed():
#     beh = DMPBehavior()
#     beh.init(3 * n_task_dims, 3 * n_task_dims)
#
#     xva = np.zeros(3 * n_task_dims)
#     beh.reset()
#     while beh.can_step():
#         eval_loop(beh, xva)
#
#     last_x = xva[:n_task_dims].copy()
#
#     eval_loop(beh, xva)
#
#     assert_array_equal(xva[:n_task_dims], last_x)
#     assert_array_equal(xva[n_task_dims:-n_task_dims], np.zeros(n_task_dims))
#     assert_array_equal(xva[-n_task_dims:], np.zeros(n_task_dims))

# def test_dmp_learn_from_demo():
#     x0, g, execution_time, dt = np.zeros(1), np.ones(1), 1.0, 0.001
#
#     beh = DMPBehavior(execution_time, dt, 20)
#     beh.init(3, 3)
#     beh.set_meta_parameters(["y0", "g"], [x0, g])
#
#     X_demo = make_minimum_jerk(x0, g, execution_time, dt)[0]
#
#     # Without regularization
#     beh.LearnfromDemo(X_demo)
#     X = beh.gen_traj()[0]
#     assert_array_almost_equal(X_demo.T[0], X, decimal=2)
#
#     # With alpha > 0
#     beh.LearnfromDemo(X_demo, regularization_coeff=1.0)
#     X = beh.gen_traj()[0]
#     assert_array_almost_equal(X_demo.T[0], X, decimal=3)
#
#     # Self-imitation
#
#     beh.LearnfromDemo(X.T[:, :, np.newaxis])
#     X2 = beh.gen_traj()[0]
#
#
#
#     plt.plot(np.linspace(0, execution_time + 0.01, 1001), X, 'r',
#              np.linspace(0, execution_time + 0.01, 1001), X2, 'b')
#     plt.show()
#
#     assert_array_almost_equal(X2, X, decimal=3)



# def test_dmp_Learn_From_Demo_2d():
#     x0, g, execution_time, dt = np.zeros(2), np.array([-1, 2], dtype=np.double), 1.0, 0.001
#
#     beh = DMPBehavior(execution_time, dt, 20)
#     beh.init(6, 6)
#     beh.set_meta_parameters(["y0", "g"], [x0, g])
#
#     X_demo = make_minimum_jerk(x0, g, execution_time, dt)[0]
#
#     # Without regularization
#     beh.LearnfromDemo(X_demo)
#     X = beh.gen_traj()[0]
#     #assert_array_almost_equal(X_demo.T[0], X, decimal=2)
#
#     # With alpha > 0
#     beh.LearnfromDemo(X_demo, regularization_coeff=1.0)
#     X = beh.gen_traj()[0]
#     #assert_array_almost_equal(X_demo.T[0], X, decimal=3)
#
#     # Self-imitation
#     beh.LearnfromDemo(X.T[:, :, np.newaxis])
#     X2 = beh.gen_traj()[0]
#
#
#     print X.shape
#     print X2. shape
#     f, axarr = plt.subplots(2, 2)
#     axarr[0, 0].plot(np.linspace(0, execution_time + 0.01, 1001),X[:,0])
#     axarr[0, 1].plot(np.linspace(0, execution_time + 0.01, 1001),X2[:, 0])
#     axarr[1, 0].plot(np.linspace(0, execution_time + 0.01, 1001), X[:, 1])
#     axarr[1, 1].plot(np.linspace(0, execution_time + 0.01, 1001), X2[:, 1])
#     plt.show()
#     assert_array_almost_equal(X2, X, decimal=3)


def test_dmp_save_and_load():
    beh_original = DMPBehavior(execution_time=0.853, dt=0.001, n_features=10)
    beh_original.init(3 * n_task_dims, 3 * n_task_dims)

    x0 = np.ones(n_task_dims) * 1.29
    g = np.ones(n_task_dims) * 2.13
    beh_original.set_meta_parameters(["y0", "g"], [x0, g])

    xva = np.zeros(3 * n_task_dims)
    xva[:n_task_dims] = x0

    beh_original.reset()
    t = 0
    while beh_original.can_step():
        eval_loop(beh_original, xva)
        if t == 0:
            assert_array_almost_equal(xva[:n_task_dims], x0)
        t += 1
    assert_array_almost_equal(xva[:n_task_dims], g, decimal=3)
    assert_equal(t, 854)
    assert_equal(beh_original.get_n_params(), n_task_dims * 10)


    beh_original.save("tmp_dmp_model.yaml")
    beh_original.save_config("tmp_dmp_config.yaml")

    beh_loaded = DMPBehavior(yaml_config="tmp_dmp_model.yaml")
    beh_loaded.init(3 * n_task_dims, 3 * n_task_dims)
    beh_loaded.load_config("tmp_dmp_config.yaml")

    if os.path.exists("tmp_dmp_model.yaml"):
            os.remove("tmp_dmp_model.yaml")
    if os.path.exists("tmp_dmp_config.yaml"):
            os.remove("tmp_dmp_config.yaml")

    xva = np.zeros(3 * n_task_dims)
    xva[:n_task_dims] = x0

    beh_loaded.reset()
    t = 0
    while beh_loaded.can_step():
        eval_loop(beh_loaded, xva)
        if t == 0:
            assert_array_almost_equal(xva[:n_task_dims], x0)
        t += 1
    assert_array_almost_equal(xva[:n_task_dims], g, decimal=3)
    assert_equal(t, 854)
    assert_equal(beh_loaded.get_n_params(), n_task_dims * 10)

def make_minimum_jerk(start, goal, execution_time=1.0, dt=0.01):
    """Create a minimum jerk trajectory.
    A minimum jerk trajectory from :math:`x_0` to :math:`g` minimizes
    the third time derivative of the positions:
    .. math::
        \\arg \min_{x_0, \ldots, x_T} \int_{t=0}^T \dddot{x}(t)^2 dt
    The trajectory will have
    .. code-block:: python
        n_steps = 1 + execution_time / dt
    steps because we start at 0 seconds and end at execution_time seconds.
    Parameters
    ----------
    start : array-like, shape (n_task_dims,)
        Initial state
    goal : array-like, shape (n_task_dims,)
        Goal
    execution_time : float, optional (default: 1)
        Execution time in seconds
    dt : float, optional (default: 0.01)
        Time between successive steps in seconds
    Returns
    -------
    X : array, shape (n_task_dims, n_steps, 1)
        The positions of the trajectory
    Xd : array, shape (n_task_dims, n_steps, 1)
        The velocities of the trajectory
    Xdd : array, shape (n_task_dims, n_steps, 1)
        The accelerations of the trajectory
    """
    x0 = np.asarray(start)
    g = np.asarray(goal)
    if x0.shape != g.shape:
        raise ValueError("Shape of initial state %s and goal %s must be equal"
                         % (x0.shape, g.shape))

    n_task_dims = x0.shape[0]
    n_steps = 1 + int(execution_time / dt)

    X = np.zeros((n_task_dims, n_steps, 1))
    Xd = np.zeros((n_task_dims, n_steps, 1))
    Xdd = np.zeros((n_task_dims, n_steps, 1))

    x = x0.copy()
    xd = np.zeros(n_task_dims)
    xdd = np.zeros(n_task_dims)

    X[:, 0, 0] = x
    tau = execution_time
    for t in range(1, n_steps):
        tau = execution_time - t * dt

        if tau >= dt:
            dist = g - x

            a1 = 0
            a0 = xdd * tau ** 2
            v1 = 0
            v0 = xd * tau

            t1 = dt
            t2 = dt ** 2
            t3 = dt ** 3
            t4 = dt ** 4
            t5 = dt ** 5

            c1 = (6. * dist + (a1 - a0) / 2. - 3. * (v0 + v1)) / tau ** 5
            c2 = (-15. * dist + (3. * a0 - 2. * a1) / 2. + 8. * v0 +
                  7. * v1) / tau ** 4
            c3 = (10. * dist + (a1 - 3. * a0) / 2. - 6. * v0 -
                  4. * v1) / tau ** 3
            c4 = xdd / 2.
            c5 = xd
            c6 = x

            x = c1 * t5 + c2 * t4 + c3 * t3 + c4 * t2 + c5 * t1 + c6
            xd = (5. * c1 * t4 + 4 * c2 * t3 + 3 * c3 * t2 + 2 * c4 * t1 + c5)
            xdd = (20. * c1 * t3 + 12. * c2 * t2 + 6. * c3 * t1 + 2. * c4)

        X[:, t, 0] = x
        Xd[:, t, 0] = xd
        Xdd[:, t, 0] = xdd

    return X, Xd, Xdd