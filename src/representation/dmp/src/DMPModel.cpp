/*************************************************************************
	> File Name: DMPModel.cpp
	> Author: Yi Zheng 
	> Mail: hczhengcq@gmail.com
	> Created Time: Thu 26 Apr 2018 07:47:57 AM PDT
 ************************************************************************/

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <DMP/DMPModel.h>
using namespace std;
namespace dmp_cpp
{
    DMPModel::DMPModel(const string& yamlFile, const string& name)
    {
        if (!from_yaml_file(yamlFile, name))
        {
            stringstream ss;
            ss << "DMPModel Error: Unable to load dmp model file from yaml: " << yamlFile;
            throw std::runtime_error(ss.str());
        }
    }

    void DMPModel::to_yaml_file(std::string filepath)
    {
        YAML::Emitter out;
        out << YAML::BeginDoc;
        out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << model_name;
        out << YAML::Key << "rbf_centers" << YAML::Value << YAML::Flow << rbf_centers;
        out << YAML::Key << "rbf_widths" << YAML::Value << YAML::Flow << rbf_widths;
        out << YAML::Key << "ts_alpha_z" << YAML::Value << ts_alpha_z;
        out << YAML::Key << "ts_beta_z" << YAML::Value << ts_beta_z;
        out << YAML::Key << "ts_tau" << YAML::Value << ts_tau;
        out << YAML::Key << "cs_execution_time" << YAML::Value << cs_execution_time;
        out << YAML::Key << "cs_alpha" << YAML::Value << cs_alpha;
        out << YAML::Key << "cs_dt" << YAML::Value << cs_dt;
        out << YAML::Key << "ts_dt" << YAML::Value << ts_dt;
        out << YAML::Key << "ft_weights" << YAML::Value << YAML::Flow << ft_weights;
        out << YAML::EndMap;
        out << YAML::EndDoc;

        ofstream fout(filepath.c_str(), ios::out | ios::app);
        fout << out.c_str();
        fout.close();
    }

    bool DMPModel::is_valid() const
    {
        stringstream ss;
        bool valid = true;
        if(cs_dt != ts_dt)
        {
            valid = false;
            ss << "Model is not valid. cs_dt and ts_dt should be equal";
            throw std::runtime_error(ss.str());
        }

        if(cs_execution_time != ts_tau)
        {
            valid = false;
            ss << "Model is not valid. cs_execution_time and ts_tau should be equal";
            throw std::runtime_error(ss.str());
        }

        //assert that execution time is more or less divisible by dt
        const int numPhases(int(cs_execution_time/cs_dt + 0.5) + 1);
        if(abs(((numPhases -1) * cs_dt) - cs_execution_time) >= 0.05)
        {
            valid = false;
            ss << "Model is not valid. cs_execution_time should be divisible by cs_dt";
            throw std::runtime_error(ss.str());
        }

        //assert that there are no nan or inf numbers anywhere
        for(unsigned i = 0; i < ft_weights.size(); ++i)
        {
            if(containsNanOrInf(ft_weights[i]))
            {
                valid = false;
                ss << "Model is not valid. Weights contain NaN or Inf";
                throw std::runtime_error(ss.str());
            }
        }

        if(containsNanOrInf(rbf_centers))
        {
            valid = false;
            ss << "Model is not valid. rbf_centers contain NaN or Inf";
            throw std::runtime_error(ss.str());
        }

        if(containsNanOrInf(rbf_widths))
        {
            valid = false;
            ss << "Model is not valid. rbf_widths contain NaN or Inf";
            throw std::runtime_error(ss.str());
        }

        return valid;
    }

    bool DMPModel::containsNanOrInf(const vector<double>& data) const
    {
        for(unsigned i = 0; i < data.size(); ++i)
        {
            if(std::isnan(data[i]) || std::isinf(data[i]))
            {
                return true;
            }
        }
        return false;
    }

    bool DMPModel::from_yaml_file(const std::string& filepath, const string& name)
    {
        ifstream fin(filepath.c_str());
        if (!fin.is_open())
        {
            stringstream ss;
            ss << "can open yaml file: " << filepath;
            throw std::runtime_error(ss.str());
        }
        return from_yaml_istream(fin, name);
    }

    bool DMPModel::from_yaml_string(const std::string& yaml, const string& name)
    {
        stringstream ss(yaml);
        return from_yaml_istream(ss, name);
    }

    bool DMPModel::from_yaml_istream(istream& stream, string name)
    {
        string name_buf;

        vector<YAML::Node> all_docs = YAML::LoadAll(stream);
        for (size_t i = 0; i < all_docs.size(); i++)
        {
            YAML::Node doc = all_docs[i];
            name_buf = doc["name"].as<string>();

            std::cout << name_buf << std::endl;

            if (name == "")
            {
                name = name_buf;
            }

            if (name_buf != name)
            {
                continue;
            }
             
            model_name = name_buf;
            rbf_centers = doc["rbf_centers"].as<vector<double> >();
            rbf_widths = doc["rbf_widths"].as<vector<double> >();;
            ts_alpha_z = doc["ts_alpha_z"].as<double>();
            ts_beta_z = doc["ts_beta_z"].as<double>();
            ts_tau = doc["ts_tau"].as<double>();
            ts_dt = doc["ts_dt"].as<double>();
            cs_execution_time = doc["cs_execution_time"].as<double>();
            cs_alpha = doc["cs_alpha"].as<double>();
            cs_dt = doc["cs_dt"].as<double>();
            ft_weights = doc["ft_weights"].as<vector<vector<double> > >();

            const double remainder = fmod(cs_execution_time, cs_dt);
            if(remainder > 0.01)
            {
                const int numPhases(int(cs_execution_time/cs_dt + 0.5) + 1);
                cs_dt = cs_execution_time/(numPhases - 1);
                ts_dt = cs_dt;
                //see http://stackoverflow.com/questions/4738768/printing-double-without-losing-precision
                //to understand why the doubles are printed in this complicated way
                cerr << "WARNING: execution time is not divisible by dt." << endl
                     << "execution time: " << fixed << scientific << setprecision(numeric_limits<double>::digits10 + 1) << cs_execution_time << endl
                     << "dt: " << fixed << scientific << setprecision(numeric_limits<double>::digits10 + 1) << cs_dt << endl
                     << "remainder: " << fixed << scientific << setprecision(numeric_limits<double>::digits10 + 1) << remainder << endl
                     << "Setting dt to: " << fixed << scientific << setprecision(numeric_limits<double>::digits10 + 1) << cs_dt << endl
                     << "If the remainder is very small, i.e. below 0.2, this is most likely caused by a loss-of-precision bug in yaml-cpp." << endl
                     << "If, however, the remainder is larger you should change your parameters." << endl;
            }

            return is_valid();
        }
    }

}

