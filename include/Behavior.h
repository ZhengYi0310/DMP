/*************************************************************************
	> File Name: Behavior.h
	> Author: Yi Zheng 
	> Mail: hczhengcq@gmail.com
	> Created Time: Sat 05 May 2018 09:51:56 AM PDT
 ************************************************************************/

#ifndef _BEHAVIOR_H
#define _BEHAVIOR_H

#include <stdexcept>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace DMP_CPP
{
    /**
     * @class Behavior 
     * Behavior interface 
     * A Behavior maps input (e.g, state) to output (e.g. next state, state difference, or action)
     */
    class Behavior 
    {
        public:
            Behavior() : numInput(-1), numOutput(-1) {}
            virtual ~Behavior() {}

            /**
             * Initialize the behavior interface 
             * \param numInputs number of the inputs
             * \param numOutputs number of the outputs
             */
            virtual void init(int numInputs, int numOutputs)
            {
                numInputs_  = numInputs;
                numOutputs_ = numOutputs;
            }

            /**
             * Clone behavior
             * throw std::run_time error if not overwriten by subclass 
             * return cloned behvior, has to be deleted 
             */
            virtual Behavior* clone()
            {
                std::cerr <<  "Used" << "\"Behavior\""  << "implementation has no" <<  "\"clone()\"" << "!" << std::endl;
                throw std::runtime_error("Used \"Behavior\" implementation has no \"clone()\"!");

            }

            /**
             * Set input for the next step.
             * If the input vector consists of positions and derivatives of these,
             * by convention all positions and all derivatives should be stored
             * contiguously.
             * \param values inputs e.g. current state of the system
             * \param numInputs number of inputs
             */
            virtual void setInputs(const double *values, int numInputs) = 0;

            /**
             * Get outputs of the last step.
             * If the output vector consists of positions and derivatives of these,
             * by convention all positions and all derivatives should be stored
             * contiguously.
             * \param[out] values outputs, e.g. desired state of the system
             * \param numOutputs expected number of outputs
             */
            virtual void getOutputs(double *values, int numOutputs) = 0;

            /*
             * Get number of inputs 
             * \return number of the inputs 
             */
            inline int getNumInputs() const {return numInputs_;}

            /*
             * Get number of outputs 
             * \return number of the outputs 
             */
            inline int getNumoutputs() const {return numOutputs_;}

            /**
             * Compute output for the received input 
             * Uses the inputs and meta-parameters to compute the outputs 
             */
            virtual void step() = 0;

            /**
             * Returns if step() can be called again.
             * \return False if the Behavior has finished executing, i.e. subsequent
             *         calls to step() will result in undefined behavior.
             *         True if the Behavior can be executed for at least one more step,
             *         i.e. step() can be called at least one more time.
             *         The default implementation always returns true.
             */
            virtual bool canStep() const { return true;}

            /**
             * Meta-parameters could be the goal, obstacles, etc.
             * Each parameter is a list of doubles identified by a key.
             */
            typedef std::map<std::string, std::vector<double> > MetaParameters;

            /**
             * Set meta-parameters.
             * Meta-parameters could be the goal, obstacles, ...
             * \throw std::runtime_error if not overwritten by subclass
             * \param params meta-parameters
             */
            virtual void setMetaParameters(const MetaParameters &params) 
            {
                throw std::runtime_error("Used \"Behavior\" implementation has no " "\"setMetaParameters()\"!");
            }

        protected:
            inline void setNumInputs(const int inputs) {numInputs_ = inputs;}
            inline void setNumOutputs(const int outputs) {numOutputs_ = outputs;}

            int numInputs_, numOutputs_;
    };
}

#endif
