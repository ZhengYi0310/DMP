import os
import numpy as np

##hackey way
import sys
sys.path.append('/home/yzheng/PycharmProjects/DMP/DMP')

from csdmp_behavior import CartesianSpaceDMPBehavior
from dmp_behavior import DMPBehavior
from nose.tools import assert_equal, assert_raises_regexp
from numpy.testing import assert_array_equal, assert_array_almost_equal
import matplotlib.pyplot as plt

CURRENT_PATH = os.sep.join(__file__.split(os.sep)[:-1])
CSDMP_CONFIG_FILE = CURRENT_PATH + os.sep + "cs_dmp_model.yaml"
if not CURRENT_PATH:
    CSDMP_CONFIG_FILE = "dmp_model.yaml"

zeroq = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def eval_loop(beh, xva):
    beh.set_inputs(xva)
    beh.step()
    beh.get_outputs(xva)


# def test_csdmp_dimensions_do_not_match():
#     beh = CartesianSpaceDMPBehavior()
#     assert_raises_regexp(ValueError, "For Cartesian space DMP, dimensionality of inputs can only be 7! --- got {} instead.".format(6),
#                          beh.init, 6, 7)
#     assert_raises_regexp(ValueError, "For Cartesian space DMP, dimensionality of outputs can only be 7! --- got {} instead.".format(6),
#     beh.init, 7, 6)
#
#
# def test_csdmp_default_dmp():
#     beh = CartesianSpaceDMPBehavior()
#     beh.init(7, 7)
#
#     x = np.copy(zeroq)
#     beh.reset()
#     t = 0
#     while beh.can_step():
#         eval_loop(beh, x)
#         t += 1
#     assert_equal(t, 1001)
#     assert_array_equal(x, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
#
#
# def test_csdmp_from_config():
#     beh = CartesianSpaceDMPBehavior(yaml_config=CSDMP_CONFIG_FILE)
#     beh.init(7, 7)
#
#     x = np.copy(zeroq)
#     beh.reset()
#     t = 0
#     while beh.can_step():
#         eval_loop(beh, x)
#         t += 1
#     assert_equal(t, 301)
#
# def test_csdmp_metaparameter_not_permitted():
#     beh = CartesianSpaceDMPBehavior()
#     beh.init(7, 7)
#     assert_raises_regexp(ValueError, "Meta parameter .* is not allowed",
#                          beh.set_meta_parameters, ["unknown"], [None])



# def test_csdmp_change_goal_velocity():
#     dt = 0.002
#     beh = CartesianSpaceDMPBehavior(dt=dt)
#     beh.init(7, 7)
#
#     beh.set_meta_parameters(["gd"], [np.ones(3)])
#
#     x = np.copy(zeroq)
#     beh.reset()
#     while beh.can_step():
#         x_prev = np.copy(x)
#         eval_loop(beh, x)
#
#     v = (x[:3] - x_prev[:3]) / dt
#
#     assert_array_almost_equal(v, np.ones(3), decimal=2)
#
# def test_csdmp_change_execution_time():
#     beh = CartesianSpaceDMPBehavior()
#     beh.init(7, 7)
#
#     X1 = beh.gen_traj()
#     beh.set_meta_parameters(["execution_time"], [2.0])
#     X2 = beh.gen_traj()
#     for i in range(6):
#         assert_equal(X2[i].shape[0], 2001)
#         assert_array_almost_equal(X1[i], X2[i][::2])
#
# def test_csdmp_change_weights():
#     beh = CartesianSpaceDMPBehavior()
#     beh.init(7, 7)
#
#     beh.set_params(np.ones(50 * 6))
#
#     x = np.copy(zeroq)
#     beh.reset()
#     while beh.can_step():
#         eval_loop(beh, x)
#
#     assert_array_almost_equal(x, zeroq, decimal=3)
#
# def test_csdmp_set_meta_params_before_init():
#     beh = CartesianSpaceDMPBehavior()
#
#     x0 = np.ones(3) * 0.43
#     g = np.ones(3) * -0.21
#     gd = np.ones(3) * 0.12
#
#     beh.set_meta_parameters(["y0", "g", "gd"], [x0, g, gd])
#     beh.init(7, 7)
#
#     x = np.zeros(7)
#     x[:3] = x0
#
#     beh.reset()
#     t = 0
#     while beh.can_step():
#         eval_loop(beh, x)
#         t += 1
#
#     assert_array_almost_equal(x[:3], g, decimal=3)
#
#
# def test_csdmp_more_steps_than_allowed():
#     beh = CartesianSpaceDMPBehavior()
#     beh.init(7, 7)
#
#     x = np.copy(zeroq)
#     beh.reset()
#     while beh.can_step():
#         eval_loop(beh, x)
#
#     last_x = x.copy()
#     eval_loop(beh, x)
#
#     assert_array_equal(x, last_x)


def test_csdmp_imitate():
    x0, g, execution_time, dt = np.zeros(3), np.ones(3), 1.0, 0.001

    beh = CartesianSpaceDMPBehavior(execution_time, dt, 20)
    beh.init(7, 7)
    beh.set_meta_parameters(["y0", "g"], [x0, g])

    X_demo = make_minimum_jerk(x0, g, execution_time, dt)[0]
    X_rot = np.tile(zeroq[3:], (X_demo.shape[1], 1)).T
    X_demo = np.vstack((X_demo[:, :, 0], X_rot))[:, :, np.newaxis][:, :, 0]


    # Without regularization
    beh.LearnfromDemo(X_demo.T)

    X = beh.gen_traj()
    X_reproduce = np.hstack((X[0], X[3]))
    assert_array_almost_equal(X_demo.T, X_reproduce, decimal=2)

    # With alpha > 0
    beh.LearnfromDemo(X_reproduce, regularization_coeff=1.0)
    X = beh.gen_traj()
    X_reproduce_1 = np.hstack((X[0], X[3]))
    assert_array_almost_equal(X_demo.T, X_reproduce, decimal=2)

    # Self-imitation
    beh.LearnfromDemo(X_reproduce_1)
    X = beh.gen_traj()
    X_reproduce_2 = np.hstack((X[0], X[3]))
    assert_array_almost_equal(X_reproduce_2, X_reproduce, decimal=2)

    # f, axarr = plt.subplots(3, 1)
    plt.figure(0)
    plt.plot(np.linspace(0, execution_time + 0.001, 1001), X_demo[0:3,:].T)
    plt.ylim(-1, 1.5)
    plt.legend(['1', '2', '3', '4'])
    plt.show()
    plt.figure(1)
    plt.plot(np.linspace(0, execution_time + 0.001, 1001),X_reproduce[:, 0:3])
    plt.legend(['1', '2', '3', '4'])
    plt.ylim(-1, 1.5)

    plt.show()

    plt.figure(2)
    plt.plot(np.linspace(0, execution_time + 0.001, 1001), X_reproduce_1[:, 0:3])
    plt.legend(['1', '2', '3', '4'])
    plt.ylim(-1, 1.5)

    plt.show()

    plt.figure(3)
    plt.plot(np.linspace(0, execution_time + 0.001, 1001), X_reproduce_2[:, 0:3])
    plt.legend(['1', '2', '3', '4'])
    plt.ylim(-1, 1.5)

    plt.show()

# def test_csdmp_change_goal():
#     beh = CartesianSpaceDMPBehavior(dt=0.01)
#     beh_d = DMPBehavior(dt=0.01)
#     beh.init(7, 7)
#     beh_d.init(9, 9)
#
#     g = np.hstack((np.ones(3), np.array([np.pi, 1.0, 0.0, 0.0])))
#     g[3:] /= np.linalg.norm(g[3:])
#     beh.set_meta_parameters(["g", "qg"], [g[:3], g[3:]])
#     beh_d.set_meta_parameters(["g"], [g[:3]])
#     x_copy = []
#     x = np.copy(zeroq)
#     beh.reset()
#     while beh.can_step():
#         eval_loop(beh, x)
#         x_copy.append(x.copy())
#     # assert_array_almost_equal(x, g, decimal=3)
#
#     x_copy = np.array(x_copy)
#
#     print x
#     for i in range(0, x_copy.shape[0]):
#                 angle = i * 0.001 * np.pi
#                 x_copy[i, 3:] = np.array([np.cos(angle  / 2.01),
#                      np.sqrt(0.5) * np.sin(angle / 2.01),
#                      0.5 * np.sin(angle / 2.01),
#                      0.5 * np.sin(angle / 2.01)])
#
#     g = np.hstack((np.ones(3), x_copy[-1, 3:]))
#     beh.set_meta_parameters(["g", "qg", "q0"], [g[:3], g[3:], x_copy[0, 3:]])
#
#     beh.LearnfromDemo(x_copy)
#     x_copy_re = beh.gen_traj()
#
#     X_reproduce_1 = np.hstack((x_copy_re[0], x_copy_re[3]))
#
#
#
#     beh_d.reset()
#     beh_d.LearnfromDemo(x_copy.T[:3, :, np.newaxis])
#     x_copy_re_d = beh_d.gen_traj()
#
#     X_reproduce_1_d =x_copy_re_d[0]
#
#
#     plt.plot(np.linspace(0, beh.execution_time + 0.01, 101),  X_reproduce_1[:, :3] )
#     plt.ylim(-0.5, 1.5)
#     plt.legend(['1', '2', '3', '4'])
#     plt.show()
#     assert (None != None)

def test_csdmp_save_and_load():
    beh_original = CartesianSpaceDMPBehavior(
        execution_time=0.853, dt=0.001, n_features=10)
    beh_original.init(7, 7)

    x0 = np.array([1.27, 3.41, 2.72])
    q0 = np.array([1.23, 2.33, 8.32, 9.29])
    q0 /= np.linalg.norm(q0)
    g = np.array([3.21, 9.34, 2.93])
    qg = np.array([2.19, 2.39, 2.94, 9.32])
    qg /= np.linalg.norm(qg)
    beh_original.set_meta_parameters(
        ["y0", "q0", "g", "qg"],
        [x0, q0, g, qg])

    x = np.hstack((x0, q0))
    beh_original.reset()
    t = 0
    while beh_original.can_step():
        eval_loop(beh_original, x)
        if t == 0:
            assert_array_almost_equal(x, np.hstack((x0, q0)))
        t += 1
    assert_array_almost_equal(x, np.hstack((g, qg)), decimal=2)
    assert_equal(t, 854)
    assert_equal(beh_original.get_n_params(), 6 * 10)

    try:
        beh_original.save("csdmp_tmp.yaml")
        beh_original.save_config("tmp_csdmp_config.yaml")

        beh_loaded = CartesianSpaceDMPBehavior(yaml_config="csdmp_tmp.yaml")
        beh_loaded.init(7, 7)
        beh_loaded.load_config("tmp_csdmp_config.yaml")
    finally:
        if os.path.exists("csdmp_tmp.yaml"):
            os.remove("csdmp_tmp.yaml")
        if os.path.exists("tmp_csdmp_config.yaml"):
            os.remove("tmp_csdmp_config.yaml")

    x = np.hstack((x0, q0))
    beh_loaded.reset()
    t = 0
    while beh_loaded.can_step():
        eval_loop(beh_loaded, x)
        if t == 0:
            assert_array_almost_equal(x, np.hstack((x0, q0)))
        t += 1
    assert_array_almost_equal(x, np.hstack((g, qg)), decimal=2)
    assert_equal(t, 854)
    assert_equal(beh_loaded.get_n_params(), 6 * 10)


def make_minimum_jerk(start, goal, execution_time=1.0, dt=0.001):
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


