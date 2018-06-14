/*************************************************************************
	> File Name: DMP.cpp
	> Author: Yi Zheng 
	> Mail: hczhengcq@gmail.com
	> Created Time: Thu 26 Apr 2018 10:34:40 AM PDT
 ************************************************************************/

#include "DMP.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cmath>
using namespace std;

namespace DMP 
{
    /*
     * Compute the phase value that corresponds to the current time in the DMP.
     * \param t current time, note that t is allowed to be outside of the range
     *          [start_t, goal_t]
     * \param alpha constant that defines the decay rate of the phase variable
     * \param goal_t time at the end of the DMP
     * \param start_t time at the start of the DMP
     * \return phase value (x)
     */
    const double phase(const double t,
                       const double alpha_x,
                       const double start_t, 
                       const double goal_t);


    /**
     * Calculates the gradient function for \p in, e.g. the derivation.
     * The returned gradient has the same shape as the input array.
     */
    template<typename PosType, typename VelType, typename TimeType>
    void Derivative(const PosType& input,
                    VelType& output,
                    const TimeType& time,
                    bool allow_final_velocity)
    {
        assert(input.cols() > 0);
        output.resize(input.rows(), input.cols());
        output.col(0).setZero();

        const int end = input.cols();
        for (int i = 1; i < end; i++)
        {
            output.col(i) = (input.col(i) - input.col(i - 1)) / (time(i) - time(i - 1));
        }

        if (!allow_final_velocity)
        {
            output.col(end - 1).setZero();
        }
    }

    /**
     * Compute axis-angle representation from quaternion (logarithmic map).
    */
    Eigen::Array3d qLog(const Eigen::Quaterniond& q);

    /**
     * Compute quaternion from axis-angle representation (exponential map).
    */
    Eigen::Quaterniond vecExp(const Eigen::Vector3d& angle_axis);

    typedef std::vector<Eigen::Quaternion<double>, Eigen::aligned_allocator<Eigen::Quaternion<double> > > QuaternionVector;

    template<typename VelType, typename TimeType>
    void quaternionDerivative(const QuaternionVector& rotations,
                             VelType& velocities,
                             const TimeType& time,
                             bool allow_final_velocity)
    {
        assert(velocities.rows() == 3);
        assert((size_t) velocities.cols() == rotations.size());
        assert(rotations.size() >= 2);

        // For the first element, assume gradient is zero
        velocities.col(0).setZero();

        const int end = (int) rotations.size();
        for (int i = 1; i < end; i++)
        {
            const Eigen::Quaterniond& q0 = rotations[i - 1];
            const Eigen::Quaterniond& q1 = rotations[i];
            const double dt = time(i) - time(i - 1);
            velocities.col(i) = 2 * qLog(q1 * q0.conjugate()) / dt; // Compute difference angular velocity based on two quaternions 
        }
        if (!allow_final_velocity)
        {
            velocities.col(end - 1).setZero();
        }
    }

    void solveConstraints(const double t0,
                          const double t1,
                          const Eigen::ArrayXd y0,
                          const Eigen::ArrayXd y0_d,
                          const Eigen::ArrayXd y0_dd,
                          const Eigen::ArrayXd y1,
                          const Eigen::ArrayXd y1_d,
                          const Eigen::ArrayXd y1_dd,
                          std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double,6, 1> > >& coefficients);

    const Eigen::MatrixXd rbfDesignMatrix(const Eigen::ArrayXd& T,
                                          const double alpha_x,
                                          const Eigen::ArrayXd& widths,
                                          const Eigen::ArrayXd& centers);


    /*
     * Linear regression with L2 regularization
     *
     * \param X design matrix, each column contains a sample
     * \param targets each column contains a sample 
     * \param regularization_coeff 
     * \param weights the resulting weights (return type) 
     */
    void L2Regression(const Eigen::MatrixXd& X,
                      const Eigen::ArrayXXd& targets,
                      const double regularization_coeff,
                      Eigen::Map<Eigen::ArrayXXd>& weights);

    /**
     * Apply 6 position, velocity, and acceleration constraints.
     */
    void applyConstraints(const double t, const Eigen::ArrayXd& goal_y, const double goal_t,
                          const std::vector<Eigen::Matrix<double, 6, 1>,Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > >& coefficients,
                          Eigen::ArrayXd& g, Eigen::ArrayXd& gd, Eigen::ArrayXd& gdd);

    /**
     * Determine accelerating forces of the forcing term during the demonstrated
     * trajectory.
     */
    void computeForces(const Eigen::ArrayXd& T,
                       const Eigen::ArrayXXd& Y,
                       Eigen::ArrayXXd& F,
                       const double alpha_y, const double beta_y,
                       bool allow_final_velocity);

    void computeForcesQuaternion(const Eigen::ArrayXd& T,
                                 const QuaternionVector& R,
                                 Eigen::ArrayXXd& F,
                                 const double alpha_y, const double beta_y,
                                 bool allow_final_velocity);

    const Eigen::ArrayXd rbfActivations(const double x,
                                        const Eigen::ArrayXd& widths,
                                        const Eigen::ArrayXd& centers,
                                        const bool normalized = true);

    const Eigen::ArrayXd forcingTerm(const double x,
                                     const Eigen::ArrayXXd& weights,
                                     const Eigen::ArrayXd& widths,
                                     const Eigen::ArrayXd& centers);

    double computeAlphaX(const double goal_x,
                         const double goal_t,
                         const double start_t)
    {
        if (goal_x <= 0.0)
            throw std::invalid_argument("Final phase must be > 0!");
        if (start_t >= goal_t)
            throw std::invalid_argument("Goal must be chronologically after start !");

        const double int_dt = 0.001;
        const double execution_time = goal_t - start_t;
        const int num_phases = (int)(execution_time / int_dt) + 1;

        // assert that the execution_time is approximately divisible by int_dt
        assert(abs(((num_phases -1) * int_dt) - execution_time) < 0.05);
        return (1.0 - pow(goal_x, 1.0 / (num_phases - 1))) * (num_phases - 1); // why ?
    }

    void initializeRBF(double* widths,
                       double* centers,
                       int num_widths,
                       int num_centers, 
                       const double goal_t,
                       const double start_t,
                       const double overlap,
                       const double alpha)
    {
        const int num_weights_per_dim = num_widths;
        if (num_widths <= 1)
                throw std::invalid_argument("The number of weights per dimension shoule > 1!");

        if(start_t >= goal_t)
            throw std::invalid_argument("Goal must be chronologically after start!");
        const double execution_time = goal_t - start_t;

        assert(num_weights_per_dim == num_widths);
        assert(num_weights_per_dim == num_centers);
        Eigen::Map<Eigen::ArrayXd> widths_array(widths, num_widths);
        Eigen::Map<Eigen::ArrayXd> centers_array(centers, num_centers);

        const double step = execution_time / (num_weights_per_dim - 1);

        const double logOverlap = -std::log(overlap);

        double t = start_t;
        centers_array(0) = phase(t, alpha, start_t, goal_t);
        for (int i = 1; i < num_weights_per_dim; i++)
        {
            t = i * step;
            centers_array(i) = phase(t, alpha, start_t, goal_t);

            // Choose width of RBF basis functions automatically so that the
            // RBF centered at one center has value overlap at the next center
            const double diff = centers_array(i) - centers_array(i - 1);
            widths_array(i - 1) = logOverlap / (diff * diff);
        }
        widths_array(num_weights_per_dim - 1) = widths_array(num_weights_per_dim - 2);
    }




    const double phase(const double t,
                       const double alpha_x,
                       const double start_t,
                       const double goal_t)
    {
        const double int_dt = 0.001;
        const double execution_time = goal_t - start_t;
        const double b = std::max(1 - alpha_x * int_dt / execution_time, 1e-10);
        return pow(b, (t-start_t) / int_dt);
    }

    void solveConstraints(double t0,
                          double t1,
                          Eigen::ArrayXd y0,
                          Eigen::ArrayXd y0_d,
                          Eigen::ArrayXd y0_dd,
                          Eigen::ArrayXd y1,
                          Eigen::ArrayXd y1_d,
                          Eigen::ArrayXd y1_dd,
                          std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > >& coefficients)
    {
        const double t02 = t0 * t0;
        const double t03 = t02 * t0;
        const double t04 = t03 * t0;
        const double t05 = t04 * t0;
        const double t12 = t1 * t1;
        const double t13 = t12 * t1;
        const double t14 = t13 * t1;
        const double t15 = t14 * t1;

        Eigen::Matrix<double, 6, 6> M;
        M << 1,   t0,      t02,      t03,       t04,        t05,
             0,   1,   2 * t0,   3 * t02,   4 * t03,    5 * t04,
             0,   0,       2,    6 * t0,   12 * t02,   20 * t03,
             1,   t1,      t12,      t13,       t14,        t15,
             0,   1,   2 * t1,   3 * t12,   4 * t13,    5 * t14,
             0, 0, 2, 6 * t1, 12 * t12, 20 * t13;

        // Solve M*b = y for b in each DOF separately
        Eigen::PartialPivLU<Eigen::Matrix<double, 6, 6> > luOfM(M);
        coefficients.clear();
        coefficients.reserve(y0.size());
        Eigen::Matrix<double, 6, 1> x;
        for(unsigned i = 0; i < y0.size(); ++i)
        {
            x << y0[i], y0_d[i], y0_dd[i], y1[i], y1_d[i], y1_dd[i];
            coefficients.push_back(luOfM.solve(x));
        }
    }

    
    void applyConstraints(const double t,
                          const Eigen::ArrayXd& goal_y,
                          const double goal_t,
                          const std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > >& coefficients,
                          Eigen::ArrayXd& g, Eigen::ArrayXd& gd, Eigen::ArrayXd& gdd)
    {
        if (t > goal_t)
        {
             /**For t > goal_t the polynomial should always 'pull' to the goal position.
              * But velocity and acceleration should be zero.
              * This is done to avoid diverging from the goal if the dmp is executed
              * longer than expected. */
            g = goal_y;
            gd.setZero();
            gdd.setZero();
        }
        else 
        {
            Eigen::Matrix<double, 1, 6> pos;
            Eigen::Matrix<double, 1, 6> vel;
            Eigen::Matrix<double, 1, 6> acc;
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            const double t5 = t4 * t;
            pos << 1, t, t2,    t3,     t4,      t5;
            vel << 0, 1, 2 * t, 3 * t2, 4 * t3,  5 * t4;
            acc << 0, 0, 2,     6 * t,  12 * t2, 20 * t3;

            for (int i = 0; i < g.size(); i++)
            {
                g[i] = pos * coefficients[i];
                gd[i] = vel * coefficients[i];
                gdd[i] = acc * coefficients[i];
            }
        }
    }

    void computeForces(const Eigen::ArrayXd& T,
                       const Eigen::ArrayXXd& Y,
                       Eigen::ArrayXXd& F,
                       const double alpha_y, const double beta_y,
                       bool allow_final_velocity)
    {
        assert(T.rows() == Y.cols());
        assert(T.rows() == F.cols());
        assert(Y.rows() == F.rows());
        const int num_dim = Y.rows();
        const int num_steps = Y.cols();

        Eigen::ArrayXXd Yd(num_dim, num_steps);
        Derivative(Y, Yd, T, allow_final_velocity);
        Eigen::ArrayXXd Ydd(num_dim, num_steps);
        Derivative(Yd, Ydd, T, false);

        //following code is equation (9) from [Muelling_IJRR_2013]
        Eigen::VectorXd start_y(Y.col(0));
        Eigen::VectorXd start_yd(Yd.col(0));
        Eigen::VectorXd start_ydd(Ydd.col(0));
        Eigen::VectorXd goal_y(Y.col(num_steps - 1));
        Eigen::VectorXd goal_yd(Yd.col(num_steps - 1));
        Eigen::VectorXd goal_ydd(Ydd.col(num_steps - 1));

        const double start_t = T(0);
        const double goal_t = T(num_steps - 1);

        std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > > coefficients;
        //solveConstraints(start_t, goal_t, goal_y, goal_yd, goal_ydd, goal_y, goal_yd, goal_ydd, coefficients);
        solveConstraints(start_t, goal_t, start_y, start_yd, start_ydd, goal_y, goal_yd, goal_ydd, coefficients);

        Eigen::ArrayXd g(num_dim);
        Eigen::ArrayXd gd(num_dim);
        Eigen::ArrayXd gdd(num_dim);

        const double t = goal_t - start_t;
        const double t2 = t * t; 

        for (int i = 0; i < num_steps; i++)
        {
            applyConstraints(T(i), goal_y, goal_t, coefficients, g, gd, gdd);
            F.col(i) = t2 * Ydd.col(i) - alpha_y * (beta_y * (g - Y.col(i)) + t * gd - t * Yd.col(i)) - t2 * gdd;
        }
    }

    void computeForcesQuaternion(const Eigen::ArrayXd& T,
                                 const QuaternionVector& R,
                                 Eigen::ArrayXXd& F,
                                 const double alpha_y, const double beta_y,
                                 bool allow_final_velocity)
    {
        assert((size_t)T.rows() == R.size());
        assert(T.rows() == F.cols());
        assert((size_t) F.cols() == R.size());
        const int num_steps = T.rows();

        Eigen::ArrayXXd Rd(3, num_steps);
        quaternionDerivative(R, Rd, T, allow_final_velocity);
        Eigen::ArrayXXd Rdd(3, num_steps);
        Derivative(Rd, Rdd, T, allow_final_velocity);

        const double t = T(num_steps - 1) - T(0);
        const double t2 = t * t; 

        for (int i = 0; i < num_steps; i++)
        {
            //F.col(i) = (t2 * Rdd.col(i) - alpha_y * (beta_y * (2.0 * qLog(R.back() * R[i].conjugate())) - t * Rd.col(i))) / (qLog(R.back() * R[0].conjugate())); //  (qLog(R.back() * R[0].conjugate())); //- alpha_y * (beta_y * 2 * qLog(R.back() * R[i].conjugate()) - t * Rd.col(i))) / (qLog(R.back() * R[0].conjugate())); // add current position to goal in unit time scaling
            F.col(i) = t2 * Rdd.col(i) - (alpha_y * (beta_y * 2 * qLog(R.back() * R[i].conjugate()) - t * Rd.col(i)));
        }
           
    }

    const Eigen::MatrixXd rbfDesignMatrix(const Eigen::ArrayXd& T,
                                          const double alpha_x,
                                          const Eigen::ArrayXd& widths,
                                          const Eigen::ArrayXd& centers)
    {
        Eigen::MatrixXd Phi(centers.rows(), T.rows());
        for (int i = 0; i < T.rows(); i++)
        {
            const double x = phase(T(i), alpha_x, T(0), T(T.rows() - 1));
            Phi.col(i) = rbfActivations(x, widths, centers, true) * x;
        }
        return Phi;
    }

    const Eigen::ArrayXd rbfActivations(const double x,
                                        const Eigen::ArrayXd& widths,
                                        const Eigen::ArrayXd& centers,
                                        const bool normalized)
    {
        Eigen::ArrayXd activations = (-widths * (x - centers).pow(2)).exp();
        if (normalized)
        {
            activations /= activations.sum();
        }
        return activations;
    }


    const Eigen::ArrayXd forcingTerm(const double x,
                                     const Eigen::ArrayXXd& weights,
                                     const Eigen::ArrayXd& widths,
                                     const Eigen::ArrayXd& centers)
    {
        const Eigen::ArrayXd activations = rbfActivations(x, widths, centers, true);
        return(x * weights.matrix() * activations.matrix()).array();
    }

    Eigen::Array3d qLog(const Eigen::Quaterniond& q)
    {
        const double norm = q.vec().norm();
        if (norm == 0)
            return Eigen::Array3d(0, 0, 0);
        return (q.vec().array() / norm * acos(q.w()));
    }

    Eigen::Quaterniond vecExp(const Eigen::Vector3d& angle_axis)
    {
        /*
        const double norm = angle_axis.norm();
        if (norm == 0)
            return Eigen::Quaterniond::Identity();
        else
        {
            const Eigen::Array3d vec = sin(norm) * angle_axis / norm;
            return Eigen::Quaterniond(cos(norm), vec.x(), vec.y(), vec.z());
        }
        */
        const double len = angle_axis.norm();
        if(len != 0)
        {
            const Eigen::Array3d vec = sin(len) * angle_axis / len;
            return Eigen::Quaterniond(cos(len), vec.x(), vec.y(), vec.z());
        }
        else
        {
            return Eigen::Quaterniond::Identity();
        }
    }


    void L2Regression(const Eigen::MatrixXd& X,
                      const Eigen::ArrayXXd& targets,
                      const double regularization_coeff,
                      Eigen::Map<Eigen::ArrayXXd>& weights)
    {
        const int num_outputs = weights.rows();
        const int num_features = weights.cols();

        for (int i = 0; i < num_outputs; i++)
        {
            weights.row(i) =((X * X.transpose() + Eigen::MatrixXd::Identity(num_features, num_features) * regularization_coeff).inverse() * X * targets.row(i).transpose().matrix()).transpose();
        }
    }

    /**
     * Represent trajectory as DMP.
     *
     * \note The final velocity will be calculated by numeric differentiation
     * from the data if allow_final_velocity is true. Otherwise we will assume
     * the final velocity to be zero. To reproduce the trajectory as closely as
     * possible, set the initial acceleration and velocity during execution to
     * zero, the final acceleration to zero and the final velocity to the value
     * that has been used during imitation.
     *
     * \param T time for each step of the trajectory
     * \param num_T number of steps
     * \param Y positions, contains num_T * num_dimensions entries in row-major
     *        order, i.e. the first position is located at the first num_dimensions
     *        entries of the array
     * \param num_steps number of steps
     * \param num_task_dims number of dimensions
     * \param weights weights that reproduce the trajectory (will be updated)
     * \param num_weights_per_dim number of features per dimension
     * \param num_weight_dims number of dimensions
     * \param widths widths of the radial basis functions (shared among DOFs)
     * \param num_widths number of RBFs
     * \param centers centers of the radial basis functions (shared among DOFs)
     * \param num_centers number of RBFs
     * \param regularization_coefficient can be set to solve instable problems
     *        where there are more weights that have to be learned than samples
     *        in the demonstrated trajectory (default: 1e-10)
     * \param alpha_y constant that has to be set for critical damping (default: 25)
     * \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
     * \param alpha_x decay rate of the phase variable (default: 25.0 / 3.0)
     * \param allow_final_velocity compute the final velocity from the data,
     *        otherwise we will assume it to be zero
     */
    void LearnfromDemo(const double* T,
                       int num_T,
                       const double* Y,
                       int num_steps,
                       int num_task_dims,
                       double* weights,
                       int num_weights_per_dim,
                       int num_weight_dims,
                       const double* widths,
                       int num_widths,
                       const double* centers,
                       int num_centers,
                       const double regularization_coefficient,
                       const double alpha_y,
                       const double beta_y,
                       const double alpha_x,
                       bool allow_final_velocity)
    {
        assert (num_steps == num_T);
        assert (num_weights_per_dim == num_widths);
        assert (num_weights_per_dim == num_centers);
        assert (num_task_dims == num_weight_dims);

        if (regularization_coefficient < 0.0)
        {
            throw std::invalid_argument("Regularization coefficient must be >= 0!");
        }
        else if (regularization_coefficient == 0.0 && num_weights_per_dim >= num_T)
        {
            throw std::invalid_argument("If the regularization coefficient is set to zero, the number of "
                                        "samples must be greater than number of weights per dimension. "
                                        "Otherwise this will result in an instable learning problem.");
        }

        Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
        Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

        Eigen::ArrayXd T_array = Eigen::Map<const Eigen::ArrayXd>(T, num_T);
        Eigen::ArrayXXd Y_array = Eigen::Map<const Eigen::ArrayXXd>(Y, num_task_dims, num_steps) ;
        Eigen::ArrayXXd F(num_task_dims, num_steps);

        // compute the reference forcing term.
        computeForces(T_array, Y_array, F, alpha_y, beta_y, allow_final_velocity);

        // construct the PHI design matrix.
        const Eigen::MatrixXd PHI = rbfDesignMatrix(T_array, alpha_x, widths_array, centers_array);
        //Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        //Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
        //std::cout << Y_array.format(OctaveFmt) << std::endl;

        Eigen::Map<Eigen::ArrayXXd> weights_array(weights, num_task_dims, num_weights_per_dim);
        L2Regression(PHI, F, regularization_coefficient, weights_array);
        //weights_array.setZero();
    }

    /**
     * Represent trajectory as a quaternion DMP.
     *
     * \note The final velocity will be calculated by numeric differentiation
     * from the data if allow_final_velocity is true. Otherwise we will assume
     * the final velocity to be zero. To reproduce the trajectory as closely as
     * possible, set the initial acceleration and velocity during execution to
     * zero, the final acceleration to zero and the final velocity to the value
     * that has been used during imitation.
     *
     * \param T time for each step of the trajectory
     * \param num_T number of steps
     * \param R roatation represented by quaterions, contains num_T * 4 entries in row-major
     *        order, i.e. the first position is located at the first num_dimensions
     *        entries of the array
     * \param num_steps number of steps
     * \param num_task_dims number of dimensions, shoule be 4 for this scenario
     * \param weights weights that reproduce the trajectory (will be updated)
     * \param num_weights_per_dim number of features per dimension
     * \param num_weight_dims number of dimensions
     * \param widths widths of the radial basis functions (shared among DOFs)
     * \param num_widths number of RBFs
     * \param centers centers of the radial basis functions (shared among DOFs)
     * \param num_centers number of RBFs
     * \param regularization_coefficient can be set to solve instable problems
     *        where there are more weights that have to be learned than samples
     *        in the demonstrated trajectory (default: 1e-10)
     * \param alpha_y constant that has to be set for critical damping (default: 25)
     * \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
     * \param alpha_x decay rate of the phase variable (default: 25.0 / 3.0)
     * \param allow_final_velocity compute the final velocity from the data,
     *        otherwise we will assume it to be zero
     */
    void LearnfromDemoQuaternion(const double* T,
                                 int num_T,
                                 const double* R,
                                 int num_steps,
                                 int num_task_dims,
                                 double* weights,
                                 int num_weights_per_dim,
                                 int num_weight_dims,
                                 const double* widths,
                                 int num_widths,
                                 const double* centers,
                                 int num_centers,
                                 const double regularization_coefficient,
                                 const double alpha_y,
                                 const double beta_y,
                                 const double alpha_x,
                                 bool allow_final_velocity)
    {
        assert (num_steps == num_T);
        assert (num_weights_per_dim == num_widths);
        assert (num_weights_per_dim == num_centers);
        assert (num_task_dims == 4);
        assert (num_weight_dims == 3);

        if (regularization_coefficient < 0.0)
        {
            throw std::invalid_argument("Regularization coefficient must be >= 0!");
        }
        else if (regularization_coefficient == 0.0 && num_weights_per_dim > num_T)
        {
            throw std::invalid_argument("If the regularization coefficient is set to zero, the number of "
                                        "samples must be greater than number of weights per dimension. "
                                        "Otherwise this will result in an instable learning problem.");
        }

        Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
        Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

        Eigen::ArrayXd T_array = Eigen::Map<const Eigen::ArrayXd>(T, num_T);
        Eigen::ArrayXXd R_array = Eigen::Map<const Eigen::ArrayXXd>(R, num_task_dims, num_steps);
        Eigen::ArrayXXd F(num_weight_dims, num_steps);

        QuaternionVector quaternion_vec;
        for (int i = 0; i < num_steps; i++)
        {
            Eigen::Quaterniond q(R_array(0, i), R_array(1, i), R_array(2, i), R_array(3, i));
            q.normalize();
            quaternion_vec.push_back(q);
        }

        // compute the reference forcing term.
        computeForcesQuaternion(T_array, quaternion_vec, F, alpha_y, beta_y, allow_final_velocity);
        //Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        //std::cout << F.format(HeavyFmt) << std::endl;

        // construct the PHI design matrix.
        const Eigen::MatrixXd PHI = rbfDesignMatrix(T_array, alpha_x, widths_array, centers_array);

        Eigen::Map<Eigen::ArrayXXd> weights_array(weights, num_weight_dims, num_weights_per_dim);
        L2Regression(PHI, F, regularization_coefficient, weights_array);
        //Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        //std::cout << weights_array.format(HeavyFmt) << std::endl;
        //weights_array.setZero();
    }

    /**
     * Execute one step of the DMP.
     *
     * source: Learning to select and generalize striking movements in robot table tennis,  K. Mülling, et al., 2013
     *
     * \param last_t time of last step (should equal t initially)
     * \param t current time
     * \param last_y last position
     * \param num_last_y number of dimensions
     * \param last_yd last velocity
     * \param num_last_yd number of dimensions
     * \param last_ydd last acceleration
     * \param num_last_ydd number of dimensions
     * \param y current position (will be updated)
     * \param num_y number of dimensions
     * \param yd velocity (will be updated)
     * \param num_yd number of dimensions
     * \param ydd acceleration (will be updated)
     * \param num_ydd number of dimensions
     * \param goal_y goal position
     * \param num_goal_y number of dimensions
     * \param goal_yd goal velocity
     * \param num_goal_yd number of dimensions
     * \param goal_ydd goal acceleration
     * \param num_goal_ydd number of dimensions
     * \param start_y start position
     * \param num_start_y number of dimensions
     * \param start_yd start velocity
     * \param num_start_yd number of dimensions
     * \param start_ydd start acceleration
     * \param num_start_ydd number of dimensions
     * \param goal_t time at the end of the DMP
     * \param start_t time at the start of the DMP
     * \param weights weights of the forcing term
     * \param num_weights_per_dim number of features per dimension
     * \param num_weight_dims number of dimensions
     * \param widths widths of the radial basis functions (shared among DOFs)
     * \param num_widths number of RBFs
     * \param centers centers of the radial basis functions (shared among DOFs)
     * \param num_centers number of RBFs
     * \param alpha_y constant that has to be set for critical damping (default: 25)
     * \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
     * \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
     * \param integration_dt temporal step-size that will be used to integrate the
     *        velocity and position of the trajectory from the acceleration,
     *        smaller values will require more computation but will reproduce the
     *        demonstration more accurately
     */
    void dmpPropagate(const double last_t,
                      const double t,
                      const double* last_y,
                      int num_last_y,
                      const double* last_yd,
                      int num_last_yd,
                      const double* last_ydd,
                      int num_last_ydd,
                      double* y,
                      int num_y,
                      double* yd,
                      int num_yd,
                      double* ydd,
                      int num_ydd,
                      const double* goal_y,
                      int num_goal_y,
                      const double* goal_yd,
                      int num_goal_yd,
                      const double* goal_ydd,
                      int num_goal_ydd,
                      const double* start_y,
                      int num_start_y,
                      const double* start_yd,
                      int num_start_yd,
                      const double* start_ydd,
                      int num_start_ydd,
                      const double goal_t,
                      const double start_t,
                      const double* weights,
                      int num_weights_per_dim,
                      int num_weight_dims,
                      const double* widths,
                      int num_widths,
                      const double* centers,
                      int num_centers,
                      const double alpha_y,
                      const double beta_y,
                      const double alpha_x,
                      const double integration_dt)
    {
        const int num_dimensions = num_last_y;
        if (start_t >= goal_t)
            throw std::invalid_argument("Goal must be chronologically after start!");

        assert(num_dimensions == num_last_y);
        assert(num_dimensions == num_last_yd);
        assert(num_dimensions == num_last_ydd);
        Eigen::Map<const Eigen::ArrayXd> last_y_array(last_y, num_last_y);
        Eigen::Map<const Eigen::ArrayXd> last_yd_array(last_yd, num_last_yd);
        Eigen::Map<const Eigen::ArrayXd> last_ydd_array(last_ydd, num_last_ydd);

        assert(num_dimensions == num_y);
        assert(num_dimensions == num_yd);
        assert(num_dimensions == num_ydd);
        Eigen::Map<Eigen::ArrayXd> y_array(y, num_y);
        Eigen::Map<Eigen::ArrayXd> yd_array(yd, num_yd);
        Eigen::Map<Eigen::ArrayXd> ydd_array(ydd, num_ydd);

        assert(num_dimensions == num_goal_y);
        assert(num_dimensions == num_goal_yd);
        assert(num_dimensions == num_goal_ydd);
        Eigen::Map<const Eigen::ArrayXd> goal_y_array(goal_y, num_goal_y);
        Eigen::Map<const Eigen::ArrayXd> goal_yd_array(goal_yd, num_goal_yd);
        Eigen::Map<const Eigen::ArrayXd> goal_ydd_array(goal_ydd, num_goal_ydd);

        assert(num_dimensions == num_start_y);
        assert(num_dimensions == num_start_yd);
        assert(num_dimensions == num_start_ydd);
        Eigen::Map<const Eigen::ArrayXd> start_y_array(start_y, num_start_y);
        Eigen::Map<const Eigen::ArrayXd> start_yd_array(start_yd, num_start_yd);
        Eigen::Map<const Eigen::ArrayXd> start_ydd_array(start_ydd, num_start_ydd);
        

        if (t <= start_t)
        {
            //std::cout << "case 1" << std::endl;
            y_array = start_y_array;
            yd_array = start_yd_array;
            ydd_array = start_ydd_array;
        }

        else 
        {
            //std::cout << "case 2" << std::endl;
            const double execution_time = goal_t - start_t;

        
            std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > > coefficients;
            solveConstraints(start_t, goal_t,
                             start_y_array, start_yd_array, start_ydd_array,
                             goal_y_array, goal_yd_array, goal_ydd_array,
                             coefficients);
        
            
            y_array = last_y_array;
            yd_array = last_yd_array;
            ydd_array = last_ydd_array;
            Eigen::ArrayXd g(num_y);
            Eigen::ArrayXd gd(num_y);
            Eigen::ArrayXd gdd(num_y);

            assert(num_weights_per_dim == num_widths);
            assert(num_weights_per_dim == num_centers);
            assert(num_weight_dims == num_dimensions);
            Eigen::Map<const Eigen::ArrayXXd> weights_array(weights, num_dimensions, num_weights_per_dim);
            Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
            Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

            double current_t = last_t;
            while (current_t < t)
            {   
                //std::cout << "update_y" << std::endl;
                double dt_int = integration_dt;
                if(t - current_t < dt_int)
                    dt_int = t - current_t;

                current_t += dt_int;

                const double x = phase(current_t, alpha_x, start_t, goal_t);
                const Eigen::ArrayXd f = forcingTerm(x, weights_array, widths_array, centers_array);

                applyConstraints(current_t, goal_y_array, goal_t, coefficients, g, gd, gdd);
                const double execution_time_squared = execution_time * execution_time;
            
                ydd_array = (alpha_y * (beta_y * (g - y_array) + gd * execution_time - execution_time * yd_array) + gdd * (execution_time_squared) + f) / execution_time_squared;

                y_array = y_array + dt_int * yd_array;
                yd_array = yd_array + dt_int * ydd_array;
            } 
            assert ((t - current_t) <= 10e-6);
            //Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
            //std::cout << yd_array.format(HeavyFmt) << std::endl;

        }
    }


     /**
     * Execute one step of the rotation DMP.
     *
     * source: Learning to select and generalize striking movements in robot table tennis,  K. Mülling, et al., 2013
     *
     * \param last_t time of last step (should equal t initially)
     * \param t current time
     * \param last_y last rotation
     * \param num_last_y number of dimensions should be 4
     * \param last_yd last rotational velocity
     * \param num_last_yd number of dimensions should be 3
     * \param last_ydd last rotationaal acceleration
     * \param num_last_ydd number of dimensions should be 3
     * \param y current rotation (will be updated)
     * \param num_y number of dimensions should be 4 
     * \param yd rotational velocity (will be updated)
     * \param num_yd number of dimensions should be 3
     * \param ydd rotational acceleration (will be updated)
     * \param num_ydd number of dimensions should be 3
     * \param goal_y goal rotation 
     * \param num_goal_y number of dimensions should be 4
     * \param goal_yd goal rotational velocity
     * \param num_goal_yd number of dimensions should be 3
     * \param goal_ydd goal rotational acceleration
     * \param num_goal_ydd number of dimensions should be 3
     * \param start_y start rotation 
     * \param num_start_y number of dimensions should be 4
     * \param start_yd start rotational velocity
     * \param num_start_yd number of dimensions should be 3
     * \param start_ydd start rotational acceleration
     * \param num_start_ydd number of dimensions should be 3
     * \param goal_t time at the end of the DMP
     * \param start_t time at the start of the DMP
     * \param weights weights of the forcing term
     * \param num_weights_per_dim number of features per dimension
     * \param num_weight_dims number of dimensions should be 3
     * \param widths widths of the radial basis functions (shared among DOFs)
     * \param num_widths number of RBFs
     * \param centers centers of the radial basis functions (shared among DOFs)
     * \param num_centers number of RBFs
     * \param alpha_y constant that has to be set for critical damping (default: 25)
     * \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
     * \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
     * \param integration_dt temporal step-size that will be used to integrate the
     *        velocity and position of the trajectory from the acceleration,
     *        smaller values will require more computation but will reproduce the
     *        demonstration more accurately
     */
    void dmpPropagateQuaternion(const double last_t, const double t, const double* last_y,
                                int num_last_y, const double* last_yd, int num_last_yd,
                                const double* last_ydd, int num_last_ydd, double* y,
                                int num_y, double* yd, int num_yd,
                                double* ydd, int num_ydd, const double* goal_y, int num_goal_y,
                                const double* goal_yd, int num_goal_yd, const double* goal_ydd,
                                int num_goal_ydd, const double* start_y, int num_start_y,
                                const double* start_yd, int num_start_yd, const double* start_ydd,
                                int num_start_ydd, const double goal_t, const double start_t,
                                const double* weights, int num_weights_per_dim, int num_weight_dims,
                                const double* widths, int num_widths, const double* centers, int num_centers,
                                const double alpha_y, const double beta_y, const double alpha_x,
                                const double integration_dt)
    {
        if (start_t > goal_t)
            throw std::invalid_argument("Goal must be chronologically after start!");

        assert(num_last_y == 4);
        assert(num_last_yd == 3);
        assert(num_last_ydd == 3);

        const Eigen::Quaterniond last_y_array(last_y[0], last_y[1], last_y[2], last_y[3]);
        Eigen::Map<const Eigen::ArrayXd> last_yd_array(last_yd, num_last_yd);
        Eigen::Map<const Eigen::ArrayXd> last_ydd_array(last_ydd, num_last_ydd);

        assert(num_y == 4);
        assert(num_yd == 3);
        assert(num_ydd == 3);
        
        Eigen::Quaterniond y_array(y[0], y[1], y[2], y[3]);
        Eigen::Map<Eigen::ArrayXd> yd_array(yd, num_yd);
        Eigen::Map<Eigen::ArrayXd> ydd_array(ydd, num_ydd);

        assert(num_goal_y == 4);
        assert(num_goal_yd == 3);
        assert(num_goal_ydd == 3);
    
        const Eigen::Quaterniond goal_y_array(goal_y[0], goal_y[1], goal_y[2], goal_y[3]);
        Eigen::Map<const Eigen::ArrayXd> goal_yd_array(goal_yd, num_goal_yd);
        Eigen::Map<const Eigen::ArrayXd> goal_ydd_array(goal_ydd, num_goal_ydd);

        assert(num_start_y == 4);
        assert(num_start_yd == 3);
        assert(num_start_ydd == 3);
    
        const Eigen::Quaterniond start_y_array(start_y[0], start_y[1], start_y[2], start_y[3]);
        Eigen::Map<const Eigen::ArrayXd> start_yd_array(start_yd, num_start_yd);
        Eigen::Map<const Eigen::ArrayXd> start_ydd_array(start_ydd, num_start_ydd);
        

        if (t <= start_t)
        {
            y_array = start_y_array;
            yd_array = start_yd_array;
            ydd_array = start_ydd_array;
        }

        else 
        {
            const double execution_time = goal_t - start_t;     
        
            y_array = last_y_array;
            yd_array = last_yd_array;
            ydd_array = last_ydd_array;
           

            assert(num_weights_per_dim == num_widths);
            assert(num_weights_per_dim == num_centers);
            assert(num_weight_dims == 3);
            Eigen::Map<const Eigen::ArrayXXd> weights_array(weights, num_weight_dims, num_weights_per_dim);
            Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
            Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

            double current_t = last_t;
            while (current_t < t)
            {    
                double dt_int = integration_dt;
                if(t - current_t < dt_int)
                    dt_int = t - current_t;

                current_t += dt_int;

                const double x = phase(current_t, alpha_x, start_t, goal_t);
                //std::cout << x << std::endl;
                const Eigen::ArrayXd f = forcingTerm(x, weights_array, widths_array, centers_array); 
                const double execution_time_squared = execution_time * execution_time;
            
                //ydd_array = (alpha_y * (beta_y * (2 * qLog(goal_y_array * y_array.conjugate())) - execution_time * yd_array) + f * qLog(goal_y_array * start_y_array.conjugate())) / execution_time_squared;
                ydd_array = (alpha_y * (beta_y * 2.0 * qLog(goal_y_array * y_array.conjugate())
                              - execution_time * yd_array)
                   + f)
                  / execution_time_squared; 

                y_array = vecExp(dt_int / 2 * yd_array) * y_array;
                yd_array = yd_array + dt_int * ydd_array;
            }
            assert ((t - current_t) <= 10e-6);
        }
        y[0] = y_array.w();
        y[1] = y_array.x();
        y[2] = y_array.y();
        y[3] = y_array.z(); 
    }

    void compute_gradient(const double* in,
                          int num_in_steps,
                          int num_in_dims,
                          double* out,
                          int num_out_steps,
                          int num_out_dims,
                          const double* time,
                          int num_time,
                          bool allow_final_velocity)
    {
        assert(num_in_steps == num_time);
        assert(num_out_steps == num_time);
        assert(num_in_dims == num_out_dims);
        Eigen::Map<const Eigen::ArrayXXd> in_array(in, num_in_dims, num_time);
        Eigen::Map<Eigen::ArrayXXd> out_array(out, num_out_dims, num_time);
        Eigen::Map<const Eigen::ArrayXd> time_array(time, num_time);
        Derivative(in_array, out_array, time_array, allow_final_velocity);
    }


    void compute_quaternion_gradient(const double* in,
                                     int num_in_steps,
                                     int num_in_dims,
                                     double* out,
                                     int num_out_steps,
                                     int num_out_dims,
                                     const double* time,
                                     int num_time,
                                     bool allow_final_velocity)
    {
        assert(num_in_steps == num_time);
        assert(num_out_steps == num_time);
        assert(num_in_dims == 4);
        assert(num_out_dims == 3);

        Eigen::Map<const Eigen::ArrayXXd> in_array(in, num_in_dims, num_time);
        QuaternionVector rotationsVector;
        for(int i = 0; i < num_time; ++i)
        {
            Eigen::Quaterniond q(in_array(0, i), in_array(1, i), in_array(2, i), in_array(3, i));
            q.normalize(); // has to be done to avoid nans
            rotationsVector.push_back(q);
        }
        Eigen::Map<Eigen::ArrayXXd> out_array(out, num_out_dims, num_time);
        Eigen::Map<const Eigen::ArrayXd> time_array(time, num_time);
        quaternionDerivative(rotationsVector, out_array, time_array, allow_final_velocity);
    }
}
