"""behavior interface"""
from abc import ABCMeta, abstractmethod
import inspect

def _print_obj(obj):
    if isinstance(obj, str):
        return "{}".format(obj)
    else:
        return obj

class Base(object):

    @classmethod
    def _get_arg_names(cls):
        """
        Get all parameters names of this class
        :return: a list of strings
        """
        args, varargs, kw, default = inspect.getargspec(cls.__init__)

        if varargs is not None:
            raise RuntimeError("objects should always specify their "
                               "parameters in the signature of their __init__ "
                               "(no varargs). {} doesn't follow this "
                               "convention.".format(cls))

        #remove "self"
        args.pop(0)
        args.sort()
        return args

    def get_args(self):
        """
        Get a specif parameter for the class
        :return:
        """
        return dict((key, getattr(self, key, None)) for key in self._get_arg_names())

    def __repr__(self):
        param_dict = self.get_args()
        params = ",".join(["{}={}".format(key, value) for key, value in self.get_args().iteritems()])
        return "{}({})".format(self.__class__.__name__, params)

class Behavior(Base):
    """
    Behavior interface, a behavior maps input (e.g.state) to output (state, action or state difference)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def init(self, n_inputs, n_outputs):
        """
        Initialize the behavior
        :param n_inputs: number of inputs
        :param n_outputs: number of outputs
        :return:
        """

    @abstractmethod
    def set_meta_parameters(self, keys, meta_parameters):
        """
        Set meta parameters
        :param keys: a list of string, names of the metaparameters
        :param meta_parameters: a list of double, values of the metaparameters
        :return:
        """

    @abstractmethod
    def set_inputs(self, inputs):
        """
        Set input for the next step.
        If the input vector consists of positions and derivatives of these,
        by convention all positions and all derivatives should be stored
        contiguously.

        :param inputs: array, (n_inputs, )
        :return:
        """

    @abstractmethod
    def get_outputs(self, outputs):
        """
        Get outputs for the next step.
        If the output vector consists of positions and derivatives of these,
        by convention all positions and all derivatives should be stored
        contiguously.

        :param outputs: array, (n_outputs, )
        :return:
        """

    @abstractmethod
    def step(self):
        """Compute output for the received input.
               Uses the inputs and meta-parameters to compute the outputs.
        """

    def can_step(self):
        """Returns if step() can be called again.
                Returns
                -------
                can_step : bool
                    Can we call step() again?
                """
        return True

class BehaviorTemplate(Base):
    """Behavior template interface."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_behavior(self, context):
        """Get behavior for a given context.
        Parameters
        ----------
        context : array-like, shape (n_context_dims,)
            Current context
        """

class OptimizableBehavior(Behavior):
    """Can be optimized with black box optimizer.
        A behavior that can be optimized with a black box optimizer must be
        **exactly** defined by a **fixed** number of parameters.
        Parameters
        ----------
        n_inputs : int
            Number of input components.
        n_outputs : int
            Number of output components.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_n_params(self):
        """Get number of parameters.
                Returns
                -------
                n_params : int
                    Number of parameters that will be optimized.
        """

    @abstractmethod
    def get_params(self):
        """Get current parameters.
        Returns
        -------
        params : array-like, shape = (n_params,)
            Current parameters.
        """

    @abstractmethod
    def set_params(self, params):
        """Set new parameter values.
                Parameters
                ----------
                params : array-like, shape = (n_params,)
                    New parameters.
        """

    @abstractmethod
    def reset(self):
        """Reset behavior.
                This method is usually called after setting the parameters to reuse
                the current behavior and clear its internal state.
        """