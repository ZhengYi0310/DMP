from libcpp cimport bool
cimport numpy as np
import numpy as np
cimport dmp_declarations as dmp 

cpdef computeAlphaX(double goal_x, double goal_t, double start_t):
    """
    Compute decay rate of the phase variable so that a desired phase is reached in the end
    \param goal_x desired phase value 
    \param goal_t time at the end of the DMP 
    \param start_t time at the start of the DMP 
    """
    cdef double cpp_goal_x = goal_x
    cdef double cpp_goal_t = goal_t 
    cdef double cpp_start_t = start_t 
    cdef double result = dmp.computeAlphaX(cpp_goal_x, cpp_goal_t, cpp_start_t)
    return result

cpdef initializeRBF(np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double goal_t, double start_t, double overlap, double alpha_x):
    """
    Initialize radial basis functions.
    
    \param widths widths of the RBFs, will be initialized
    \param num_widths number of RBFs
    \param centers centers of the RBFs, will be initialized
    \param num_centers number of RBFs
    \param goal_t time at the end of the DMP
    \param start_t time at the start of the DMP
    \param overlap value of each RBF at the center of the next RBF
    \param alpha decay rate of the phase variable (default: 25.0 / 3.0)
    """
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_overlap = overlap 
    cdef double cpp_alpha_x = alpha_x
    dmp.initializeRBF(&widths[0], &centers[0], widths.shape[0], centers.shape[0], cpp_goal_t, cpp_start_t,
                      cpp_overlap, cpp_alpha_x)

cpdef LearnfromDemo(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=2] Y, np.ndarray[double, ndim=2] weights,
                    np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double regularization_coefficient, double alpha_y, double beta_y, double alpha_x, bool allow_final_velocity):
    cpdef double cpp_regularization_coefficient = regularization_coefficient
    cpdef double cpp_alpha_y = alpha_y
    cpdef double cpp_beta_y = beta_y
    cpdef double cpp_alpha_x = alpha_x
    cpdef bool   cpp_allow_final_velocity = allow_final_velocity
    dmp.LearnfromDemo(&T[0], T.shape[0], &Y[0, 0], Y.shape[0], Y.shape[1], &weights[0, 0], weights.shape[0], 
                      weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0], 
                      cpp_regularization_coefficient, cpp_alpha_y, cpp_beta_y, cpp_alpha_x, cpp_allow_final_velocity)


cpdef LearnfromDemoQuaternion(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=2] R, np.ndarray[double, ndim=2] weights,
                    np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double regularization_coefficient, double alpha_y, double beta_y, double alpha_x, bool allow_final_velocity):
    cdef double cpp_regularization_coefficient = regularization_coefficient
    cdef double cpp_alpha_y = alpha_y
    cdef double cpp_beta_y = beta_y
    cdef double cpp_alpha_x = alpha_x
    cdef bool   cpp_allow_final_velocity = allow_final_velocity
    dmp.LearnfromDemoQuaternion(&T[0], T.shape[0], &R[0, 0], R.shape[0], R.shape[1], &weights[0, 0], weights.shape[0], 
                      weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0], 
                      cpp_regularization_coefficient, cpp_alpha_y, cpp_beta_y, cpp_alpha_x, cpp_allow_final_velocity)


cpdef dmpPropagate(double last_t, double t, np.ndarray[double, ndim=1] last_y, np.ndarray[double, ndim=1] last_yd, np.ndarray[double, ndim=1] last_ydd, 
                   np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] y_d, np.ndarray[double, ndim=1] y_dd, 
                   np.ndarray[double, ndim=1] goal_y, np.ndarray[double, ndim=1] goal_yd, np.ndarray[double, ndim=1] goal_ydd,
                   np.ndarray[double, ndim=1] start_y, np.ndarray[double, ndim=1] start_yd, np.ndarray[double, ndim=1] start_ydd,
                   double goal_t, double start_t, np.ndarray[double, ndim=2] weights, np.ndarray[double, ndim=1] widths, 
                   np.ndarray[double, ndim=1] centers, double alpha_y, double beta_y, double alpha_x, double integration_dt):
    cdef double cpp_last_t = last_t
    cdef double cpp_t = t 
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_alpha_y = alpha_y
    cdef double cpp_beta_y = beta_y
    cdef double cpp_alpha_x = alpha_x
    cdef double cpp_integration_dt = integration_dt
    dmp.dmpPropagate(cpp_last_t, cpp_t, &last_y[0], last_y.shape[0], &last_yd[0], last_yd.shape[0], &last_ydd[0], last_ydd.shape[0],
                     &y[0], y.shape[0], &y_d[0], y_d.shape[0], &y_dd[0], y_dd.shape[0],
                     &goal_y[0], goal_y.shape[0], &goal_yd[0], goal_yd.shape[0], &goal_ydd[0], goal_ydd.shape[0],
                     &start_y[0], start_y.shape[0], &start_yd[0], start_yd.shape[0], &start_ydd[0], start_ydd.shape[0], cpp_goal_t, cpp_start_t,
                     &weights[0, 0], weights.shape[0], weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0],
                     cpp_alpha_y, cpp_beta_y, cpp_alpha_x, cpp_integration_dt)

cpdef dmpPropagateQuaternion(double last_t, double t, np.ndarray[double, ndim=1] last_y, np.ndarray[double, ndim=1] last_yd, np.ndarray[double, ndim=1] last_ydd, 
                   np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] y_d, np.ndarray[double, ndim=1] y_dd, 
                   np.ndarray[double, ndim=1] goal_y, np.ndarray[double, ndim=1] goal_yd, np.ndarray[double, ndim=1] goal_ydd,
                   np.ndarray[double, ndim=1] start_y, np.ndarray[double, ndim=1] start_yd, np.ndarray[double, ndim=1] start_ydd,
                   double goal_t, double start_t, np.ndarray[double, ndim=2] weights, np.ndarray[double, ndim=1] widths, 
                   np.ndarray[double, ndim=1] centers, double alpha_y, double beta_y, double alpha_x, double integration_dt):
    cdef double cpp_last_t = last_t
    cdef double cpp_t = t 
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_alpha_y = alpha_y
    cdef double cpp_beta_y = beta_y
    cdef double cpp_alpha_x = alpha_x
    cdef double cpp_integration_dt = integration_dt
    dmp.dmpPropagateQuaternion(cpp_last_t, cpp_t, &last_y[0], last_y.shape[0], &last_yd[0], last_yd.shape[0], &last_ydd[0], last_ydd.shape[0],
                     &y[0], y.shape[0], &y_d[0], y_d.shape[0], &y_dd[0], y_dd.shape[0],
                     &goal_y[0], goal_y.shape[0], &goal_yd[0], goal_yd.shape[0], &goal_ydd[0], goal_ydd.shape[0],
                     &start_y[0], start_y.shape[0], &start_yd[0], start_yd.shape[0], &start_ydd[0], start_ydd.shape[0], cpp_goal_t, cpp_start_t,
                     &weights[0, 0], weights.shape[0], weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0],
                     cpp_alpha_y, cpp_beta_y, cpp_alpha_x, cpp_integration_dt)

cpdef compute_gradient(np.ndarray[double, ndim=2] _in, np.ndarray[double, ndim=2] out,
                       np.ndarray[double, ndim=1] time, bool allow_final_velocity):
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    dmp.compute_gradient(&_in[0, 0], _in.shape[0], _in.shape[1],
                         &out[0, 0], out.shape[0], out.shape[1],
                         &time[0], time.shape[0],
                         cpp_allow_final_velocity)

cpdef compute_quaternion_gradient(np.ndarray[double, ndim=2] _in, np.ndarray[double, ndim=2] out,
                       np.ndarray[double, ndim=1] time, bool allow_final_velocity):
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    dmp.compute_quaternion_gradient(&_in[0, 0], _in.shape[0], _in.shape[1],
                         &out[0, 0], out.shape[0], out.shape[1],
                         &time[0], time.shape[0],
                         cpp_allow_final_velocity)




