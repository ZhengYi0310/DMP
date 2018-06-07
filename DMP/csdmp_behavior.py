import yaml
import warnings
import StringIO
import numpy as np
from behavior import OptimizableBehavior
from dmp_behavior import save_dmp_model, load_dmp_model
import dmp

CSDMP_META_PARAMETERS = ["x0", "g", "gd", "gdd", "q0", "qg", "execution_time"]

class CartesianSpaceDMPBehavior(OptimizableBehavior):
    """
    Cartesian space dynamic movement primitive, with quaternion as rotation representation
    """
    def __init__(self, execution_time=0.0, dt=0.001, n_features=50, yaml_config=None):
        if yaml_config is None:
            self.execution_time = execution_time
            self.dt = dt
            self.n_features = n_features
        else:
            self.yaml_config = yaml_config

    def init(self, n_inputs, n_outputs):
        if n_inputs != 7:
            raise ValueError("For Cartesian space DMP, dimensionality of inputs can only be 7! --- got {} instead.".format(n_inputs))
        if n_outputs != 7:
            raise ValueError("For Cartesian space DMP, dimensionality of outputs can only be 7! --- got {} instead.".format(n_outputs))

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.n_task_dims = 6

        if hasattr(self, "yaml_config"):
            load_dmp_model(self, self.yaml_config)

        else:
            self.name = "Cartesian DMP"
            self.alpha_x = dmp.computeAlphaX(0.01, self.execution_time, 0.0)
            self.widths = np.zeros(self.n_features)
            self.centers = np.zeros(self.n_features)
            dmp.initializeRBF(self.widths, self.centers, self.execution_time, 0.0, 0.8, self.alpha_x)
            self.alpha_y = 25.0
            self.beta_y = self.alpha_y / 4.0

            self.position_weights = np.zeros((self.n_features, 3))
            self.orientation_weights = np.zeros((self.n_features, 3))
            self.weights = np.zeros((self.n_features, self.n_task_dims))

        if not hasattr(self, "y0"):
            self.y0 = None
        if not hasattr(self, "y0d"):
            self.y0d = None
        if not hasattr(self, "y0dd"):
            self.y0dd = None
        if not hasattr(self, "g"):
            self.g = None
        if not hasattr(self, "gd"):
            self.gd = None
        if not hasattr(self, "gdd"):
            self.gdd = None

        if not hasattr(self, "q0"):
            self.q0 = np.array([0.0, 1.0, 0.0, 0.0])
        if not hasattr(self, "q0d"):
            self.q0d = None
        if not hasattr(self, "q0dd"):
            self.q0dd = None
        if not hasattr(self, "qg"):
            self.qg = np.array([0.0, 1.0, 0.0, 0.0])
        if not hasattr(self, "qgd"):
            self.qgd = None
        if not hasattr(self, "qgdd"):
            self.qgdd = None

        self.reset()

    def reset(self):
        """
        Reset DMP
        :return:
        """
        if self.y0 is None:
            self.last_y = np.zeros(self.n_task_dims)
        else:
            self.last_y = np.copy(self.y0)
        self.last_yd = np.copy(self.y0d)
        self.last_ydd = np.copy(self.y0dd)

        self.y = np.zeros(3)
        self.yd = np.zeros(3)
        self.ydd = np.zeros(3)

        self.last_r = np.copy(self.q0)
        self.last_rd = np.copy(self.q0d)
        self.last_rdd = np.copy(self.q0dd)

        self.r = np.zeros(4)
        self.rd = np.zeros(3)
        self.rdd = np.zeros(3)


        self.last_t = 0.0
        self.t = 0.0

    def get_weights(self):
        return np.hstack((self.position_weights, self.orientation_weights))

    def set_weights(self, weights):
        if not hasattr(self, "position_weights"):
            self.position_weights = np.zeros((self.n_features, 3))
        if not hasattr(self, "orientation_weights"):
            self.orientation_weights = np.zeros(())

        self.position_weights[:] = weights[:, :3]
        self.orientation_weights[:] = weights[:, 3:]
    weights = property(get_weights, set_weights)

    def set_meta_parameters(self, keys, meta_parameters):
        """
        Set dmp meta prameters
        :param keys:
        :param meta_parameters:
        :return:
        """
        for key, meta_parameter in zip(keys, meta_parameters):
            if key not in CSDMP_META_PARAMETERS:
                raise ValueError(
                    "Meta parameter '%s' is not allowed, use one of %r"
                    % (key, CSDMP_META_PARAMETERS))
            setattr(self, key, meta_parameter)

    def set_inputs(self, inputs):
        self.last_y   = inputs['y']
        self.last_r   = inputs['r']
        self.last_yd  = inputs['yd']
        self.last_rd  = inputs['rd']
        self.last_rdd = inputs['rdd']
        self.last_ydd = inputs['ydd']

    def get_outputs(self, outputs):
        outputs['y']   = self.y
        outputs['r']   = self.r
        outputs['yd']  = self.yd
        outputs['rd']  = self.rd
        outputs['ydd'] = self.ydd
        outputs['rdd'] = self.rdd



    def step(self):
        """
        Compute desired position, velocity and acceleration of the transformation system
        :return:
        """
        if self.n_task_dims != 6:
            raise  ValueError("Task dimensions are not 6!")

        dmp.dmpPropagate(self.last_t, self.t,
                      self.last_y, self.last_yd, self.last_ydd,
                      self.y,      self.yd,      self.ydd,
                      self.g,      self.gd,      self.gdd,
                      self.y0,     self.y0d,     self.y0dd,
                      self.execution_time, 0.0,
                      self.position_weights, self.widths, self.centers,
                      self.alpha_y, self.beta_y, self.alpha_x,
                      self.dt)

        dmp.dmpPropagate(self.last_t, self.t,
                         self.last_r, self.last_rd, self.last_rdd,
                         self.r, self.rd, self.rdd,
                         self.qg, self.qgd, self.qgdd,
                         self.q0, self.q0d, self.q0dd,
                         self.execution_time, 0.0,
                         self.orientation_weights, self.widths, self.centers,
                         self.alpha_y, self.beta_y, self.alpha_x,
                         self.dt)

        if self.t == self.last_t:
            self.last_t = -1.0
        else:
            self.t += self.last_t

    def can_step(self):
        return self.t <= self.execution_time

    def get_n_params(self):
        """
        Get the number of weights
        :return:
        """
        return self.weights.size

    def get_params(self):
        """
        Get current weights
        :return:
        """
        return self.weights

    def set_params(self, params):
        self.weights[:, :] = np.reshape(params, (self.n_features, self.n_task_dims))

    def LearnfromDemo(self, Y, Yd= None, Ydd=None, regularization_coeff=1e-10, allow_final_velocity = False):
        """

        :param Y: array, shape (7, n_steps, n_demos) : x, y, z, w, rx, ry, rz
                  The demonstrated trajectories to be imitated.
        :param Yd: array, shape (n_task_dims, n_steps, n_demos), optional
                   Velocities of the demonstrated trajectories.
        :param Ydd: array, shape (n_task_dims, n_steps, n_demos), optional
                    Velocities of the demonstrated trajectories.
        :param regularization_coeff: Regularization coefficient for the ridge regression
        :param allow_final_velocity:
        :return:
        """
        if Y.shape[2] > 1:
            warnings.warn("Imitations only accepts one demonstration!")
        if Yd is not None:
            warnings.warn("Xd is deprecated")
        if Ydd is not None:
            warnings.warn("Xdd is deprecated")

        Y = Y[:, :, 0].T.copy()

        Y_pos = Y[:, :3]
        Y_ori = Y[:, 3:]
        dmp.LearnfromDemo(np.arange(0, self.execution_time + self.dt, self.dt),
                          Y_pos, self.position_weights, self.widths, self.centers,
                          regularization_coeff, self.alpha_y, self.beta_y, self.alpha_x,
                          allow_final_velocity)

        dmp.LearnfromDemoQuaternion(np.arange(0, self.execution_time + self.dt, self.dt),
                                    Y_ori, self.orientation_weights, self.widths, self.centers,
                                    regularization_coeff, self.alpha_y, self.beta_y, self.alpha_x,
                                    allow_final_velocity)

    def gen_traj(self):
        """
        Gnerate the trajectory represented by the DMP in the Open Loop
        :return:
        """
        last_t = 0.0
        last_y = np.copy(self.y0)
        last_yd = np.copy(self.y0d)
        last_ydd = np.copy(self.y0dd)

        last_r = np.copy(self.q0)
        last_rd = np.copy(self.q0d)
        last_rdd = np.copy(self.q0dd)

        y = np.zeros(self.n_task_dims)
        yd = np.zeros(self.n_task_dims)
        ydd = np.zeros(self.n_task_dims)

        r = np.zeros(4)
        rd = np.zeros(3)
        rdd = np.zeros(3)

        Y =  []
        Yd = []
        Ydd = []

        R = []
        Rd = []
        Rdd = []

        for t in np.arange(0, self.execution_time + self.dt, self.dt):
            dmp.dmpPropagate(last_t, t,
                             last_y, last_yd, last_ydd,
                             y, yd, ydd,
                             self.g,  self.gd,  self.gdd,
                             self.y0, self.y0d, self.y0dd,
                             self.execution_time, 0.0,
                             self.position_weights, self.widths, self.centers,
                             self.alpha_y, self.beta_y, self.alpha_x,
                             self.dt)

            dmp.dmpPropagateQuaternion(last_t, t,
                                       last_r, last_rd, last_rdd,
                                       r, rd, rdd,
                                       self.qg, self.qgd, self.qgdd,
                                       self.q0, self.q0d, self.q0dd,
                                       self.execution_time, 0.0,
                                       self.orientation_weights, self.widths, self.centers,
                                       self.alpha_y, self.beta_y, self.alpha_x,
                                       0.001)

            last_t = t

            last_r[:, :] = r
            last_rd[:, :] = rd
            last_rdd[:, :] = rdd
            last_y[:, :] = y
            last_yd[:, :] = yd
            last_ydd[:, :] = ydd

            Y.append(y.copy())
            Yd.append(yd.copy())
            Ydd.append(ydd.copy())
            R.append(y.copy())
            Rd.append(yd.copy())
            Rdd.append(ydd.copy())
        return np.array(Y), np.array(Yd), np.array(Ydd), \
               np.array(R), np.array(Rd), np.array(Rdd)

    def save_config(self, filename):
        config = {}
        config = {}
        config["name"] = self.name
        config["dmp_execution_time"] = self.execution_time

        config["dmp_startPosition"] = self.y0.tolist()
        config["dmp_startVelocity"] = self.y0d.tolist()
        config["dmp_startAcceleration"] = self.y0dd.tolist()
        config["dmp_startOrientation"] = self.q0.tolist()
        config["dmp_startAngularVelocity"] = self.q0d.tolist()

        config["dmp_endPosition"] = self.g.tolist()
        config["dmp_endVelocity"] = self.gd.tolist()
        config["dmp_endAcceleration"] = self.gdd.tolist()
        config["dmp_endOrientation"] = self.qg.tolist()


        config_content = StringIO.StringIO()
        yaml.dump(config, config_content)
        with open(filename, "w") as f:
            f.write("---\n")
            f.write(config_content.getvalue())
            f.write("---\n")
        config_content.close()

    def load_config(self, filename):
        config = yaml.load(open(filename, "r"))

        self.execution_time = config["dmp_execution_time"] =
        self.y0 = np.array(config["dmp_startPosition"], dtype=np.float)
        self.y0d = np.array(config["dmp_startVelocity"], dtype=np.float)
        self.y0dd = np.array(config["dmp_startAcceleration"], dtype=np.float)
        self.q0 = np.array(config["dmp_startOrientation"], dtype=np.float)
        self.q0d = np.array(config["dmp_startAngularVelocity"], dtype=np.float)
        self.g = np.array(config["dmp_endPosition"], dtype=np.float)
        self.gd = np.array(config["dmp_endVelocity"], dtype=np.float)
        self.gdd = np.array(config["dmp_endAcceleration"], dtype=np.float)
        self.qg = np.array(config["dmp_endOrientation"], dtype=np.float)