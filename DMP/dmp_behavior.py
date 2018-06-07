import yaml
import warnings
import StringIO
import numpy as np
from behavior import OptimizableBehavior
import dmp


DMP_META_PARAMETERS = ["x0", "g", "gd", "gdd", "execution_time"]

def load_dmp_model(dmp, filename):
    """
    load a dmp model from a yaml file.

    :param dmp: DMP object
    :param filename: Name of the YAML file
    :return:
    """
    model = yaml.load(open(filename, 'r'))
    dmp.name = model["name"]
    dmp.alpha_x = model["cs_alpha"]
    dmp.widths = np.array(model["rbf_widths"], dtype=np.float)
    dmp.centers = np.array(model["rbf_centers"], dtype=np.float)
    dmp.alpha_y = model["ts_alpha_z"]
    dmp.beta_y = model["ts_beta_z"]
    dmp.execution_time = model["ts_tau"]
    dmp.dt = model["ts_dt"]
    dmp.n_features = dmp.widths.shape[0]
    dmp.weights = np.array(model["ft_weights"], dtype=np.float).reshape(dmp.n_task_dims , dmp.n_features)

    if dmp.execution_time != model(["cs_execution_time"]):
        raise ValueError("Inconsistent execution times: {} != {}"
                         .format(model["ts_tau"], model["cs_execution_time"]))

    if dmp.dt != model["cs_dt"]:
        raise ValueError("Inconsistent time steps: {} != {}"
                         .format(model["cs_dt"], model["cs_execution_time"]))

def save_dmp_model(dmp, filename):
    """
    save a dmp model from a yaml file.
    :param dmp:
    :param filename:
    :return:
    """
    model = {}
    model["name"] = dmp.name
    model["cs_alpha"] = dmp.alpha_x
    model["cs_dt"] = dmp.dt
    model["cs_execution_time"] = dmp.execution_time
    model["rbf_widths"] = dmp.widths.tolist()
    model["rbf_centers"] = dmp.centers.tolist()
    model["ts_alpha_z"]= dmp.alpha_y
    model["ts_beta_z"] = dmp.beta_y
    model["ts_tau"] = dmp.execution_time
    model["ts_dt"] = dmp.dt
    model["ft_weights"] = dmp.weights.tolist()

    model_content = StringIO.StringIO()
    yaml.dump(model, model_content)
    with open(filename, "w") as f:
        f.write("---\n")
        f.write(model_content.getvalue())
        f.write("...\n")
    model_content.close()

class DMPBehavior(OptimizableBehavior):
    """
    Dynamic Movement Primitive

    Parameters can be optimized using a black box optimizer
    """
    def __init__(self, execution_time=1.0, dt=0.001, n_features=50, yaml_config=None):
        if yaml_config is None:
            self.execution_time = execution_time
            self.dt = dt
            self.n_features = n_features
        else:
            self.yaml_config = yaml_config

    def init(self, n_inputs, n_outputs):
        if n_inputs != n_outputs:
            raise ValueError("Input and output dimensions much match, got {} inputs and {} outputs."
                             .format(n_inputs, n_outputs))

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.n_task_dims = self.n_inputs / 3.0

        if hasattr(self, "yaml_config"):
            load_dmp_model(self, self.yaml_config)

        else:
            self.name = "DMP"
            self.alpha_x = dmp.computeAlphaX(0.01, self.execution_time, 0.0)
            self.widths = np.zeros(self.n_features)
            self.centers = np.zeros(self.n_features)
            dmp.initializeRBF(self.widths, self.centers, self.execution_time, 0.0, 0.8, self.alpha_x)
            self.alpha_y = 25.0
            self.beta_y = self.alpha_y / 4.0
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
        self.reset()

    def set_meta_parameters(self, keys, meta_parameters):
        """
        Set DMP meta prametets
        :param keys:
        :param meta_parameters:
        :return:
        """
        for key, meta_parameter in zip(keys, meta_parameters):
            if key not in DMP_META_PARAMETERS:
                raise ValueError(
                    "Meta parameter '%s' is not allowed, use one of %r"
                    % (key, DMP_META_PARAMETERS))
            setattr(self, key, meta_parameter)

    def set_inputs(self, inputs):
        """
        Set the input for the next dim, if there is no x0, use the first position as x0
        :param inputs: array, (3 * n_task_dims, )
        :return:
        """
        self.last_y = inputs[:self.n_task_dims]
        self.last_yd = inputs[self.n_task_dims + 1 : -self.n_task_dims]
        self.last_ydd = inputs[-self.n_task_dims:]

    def get_outputs(self, outputs):
        """
        Set the input for the next dim, if there is no x0, use the first position as x0
        :param inputs: array, (3 * n_task_dims, )
        :return:
        """
        outputs[:self.n_task_dims] = self.y[:]
        outputs[self.n_task_dims + 1: -self.n_task_dims] = self.yd[:]
        outputs[-self.n_task_dims:] = self.ydd[:]

    def step(self):
        """
        Compute desired position, velocity and acceleration of the transformation system
        :return:
        """
        if self.n_task_dims == 0:
            raise  ValueError("Task dimensions are 0!")

        dmp.dmpPropagate(self.last_t, self.t,
                      self.last_y, self.last_yd, self.last_ydd,
                      self.y,      self.yd,      self.ydd,
                      self.g,      self.gd,      self.gdd,
                      self.y0,     self.y0d,     self.y0dd,
                      self.execution_time, 0.0,
                      self.weights, self.widths, self.centers,
                      self.alpha_y, self.beta_y, self.alpha_x,
                      self.dt)

        if self.t == self.last_t:
            self.last_t = -1.0
        else:
            self.t += self.last_t

    def can_step(self):
        """
        check if step() can be called again.
        :return:
        """
        return (self.t <= self.execution_time)

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

        self.y = np.zeros(self.n_task_dims)
        self.yd = np.zeros(self.n_task_dims)
        self.ydd = np.zeros(self.n_task_dims)

        self.last_t = 0.0
        self.t = 0.0

    def LearnfromDemo(self, Y, Yd= None, Ydd=None, regularization_coeff=1e-10, allow_final_velocity = False):
        """

        :param Y: array, shape (n_task_dims, n_steps, n_demos)
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

        dmp.LearnfromDemo(np.arange(0, self.execution_time + self.dt, self.dt),
                          regularization_coeff, self.weights, self.widths, self.centers,
                          1e-10, self.alpha_y, self.beta_y, self.alpha_x,
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

        y = np.zeros(self.n_task_dims)
        yd = np.zeros(self.n_task_dims)
        ydd = np.zeros(self.n_task_dims)

        Y =  []
        Yd = []
        Ydd = []

        for t in np.arange(0, self.execution_time + self.dt, self.dt):
            dmp.dmpPropagate(last_t, t,
                             last_y, last_yd, last_ydd,
                             y, yd, ydd,
                             self.y0, self.y0d, self.y0dd,
                             self.execution_time, 0.0,
                             self.weights, self.widths, self.centers,
                             self.alpha_y, self.beta_y, self.alpha_x,
                             self.dt)
            last_t = t
            last_y[:, :] = y
            last_yd[:, :] = yd
            last_ydd[:, :] = ydd

            Y.append(y.copy())
            Yd.append(yd.copy())
            Ydd.append(ydd.copy())

        return np.array(Y), np.array(Yd), np.array(Ydd)

    def save_config(self, filename):
        config = {}
        config = {}
        config["name"] = self.name
        config["dmp_execution_time"] = self.execution_time
        config["dmp_startPosition"] = self.y0.tolist()
        config["dmp_startVelocity"] = self.y0d.tolist()
        config["dmp_startAcceleration"] = self.y0dd.tolist()
        config["dmp_endPosition"] = self.g.tolist()
        config["dmp_endVelocity"] = self.gd.tolist()
        config["dmp_endAcceleration"] = self.gdd.tolist()

        config_content = StringIO.StringIO()
        yaml.dump(config, config_content)
        with open(filename, "w") as f:
            f.write("---\n")
            f.write(config_content.getvalue())
            f.write("---\n")
        config_content.close()

    def load_config(self, filename):
        config = yaml.load(open(filename, "r"))

        self.execution_time = config["dmp_execution_time"]
        self.y0 = np.array(config["dmp_startPosition"], dtype=np.float)
        self.y0d = np.array(config["dmp_startVelocity"], dtype=np.float)
        self.y0dd = np.array(config["dmp_startAcceleration"], dtype=np.float)
        self.g = np.array(config["dmp_endPosition"], dtype=np.float)
        self.gd = np.array(config["dmp_endVelocity"], dtype=np.float)
        self.gdd = np.array(config["dmp_endAcceleration"], dtype=np.float)



