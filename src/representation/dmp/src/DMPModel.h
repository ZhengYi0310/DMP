/*************************************************************************
	> File Name: DMPModel.h
	> Author: Yi Zheng 
	> Mail: hczhengcq@gmail.com
	> Created Time: Thu 26 Apr 2018 07:04:47 AM PDT
 ************************************************************************/

#ifndef _DMPMODEL_H
#define _DMPMODEL_H
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <math.h>
using namespace std;
namespace dmp_cpp
{
    class DMPModel
    {
        public:
            std::string model_name;
            std::vector<double> rbf_centers;
            std::vector<double> rbf_widths;
            double ts_alpha_z;
            double ts_beta_z; // usually set to ts_alpha_z/4 for critical damping 
            double ts_tau;
            double cs_execution_time; // should always be the same as ts_tau 
            double cs_alpha;
            double cs_dt;
            double ts_dt; // ts_dt * num_phases = execution_time
            std::vector<std::vector<double> > ft_weights;

            DMPModel() {}

            /*
             * throw std::error if a DMP model is not loaded 
             */
            DMPModel(const std::string& yamlFile, const std::string& name);

            bool from_yaml_file(const std::string& filepath, const string& name);
            bool from_yaml_istream(std::istream& stream, string name);
            bool from_yaml_string(const std::string& yaml, const string& name);

            /**
             * \return true if the model is consistent, false otherwise
             */
            virtual bool is_valid() const;

            void to_yaml_file(std::string filepath);

            friend std::ostream& operator << (std::ostream& os, DMPModel& model);

        private:
            /*
             * return true if the vector contains at least one number that is NaN or Inf
             */
            bool containsNanOrInf(const std::vector<double>& data) const;
    };

    template<typename T>
    inline std::ostream& operator<<(std::ostream& stm, const std::vector<T>& obj)
    {
        stm << "[";
        if (!obj.empty())
        {
            for (size_t i = 0; i < obj.size() - 1; i++)
            {
                stm << obj[i] << ", ";
            }
            stm << obj.back();
        }
        stm << "]";
        return stm;
    }

    inline std::ostream& operator<<(std::ostream& os, DMPModel& model)
    {
        os << "rbf_centers: " << model.rbf_centers << std::endl;
        os << "rbf_widths: " << model.rbf_widths << std::endl;
        os << "ts_alpha_z: " << model.ts_alpha_z << std::endl;
        os << "ts_beta_z: " << model.ts_beta_z << std::endl;
        os << "ts_tau: " << model.ts_tau << std::endl;
        os << "ts_dt: " << model.ts_dt << std::endl;
        os << "cs_execution_time: " << model.cs_execution_time << std::endl;
        os << "cs_alpha: " << model.cs_alpha << std::endl;
        os << "cs_dt: " << model.cs_dt << std::endl;
        os << "ft_weights: " << model.ft_weights << std::endl;
        return os;
    }
}


#endif
