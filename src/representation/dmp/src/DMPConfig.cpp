/*************************************************************************
	> File Name: DMPConfig.cpp
	> Author: Yi Zheng 
	> Mail: hczhengcq@gmail.com
	> Created Time: Thu 26 Apr 2018 02:31:49 AM PDT
 ************************************************************************/

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <DMP/DMPConfig.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
namespace dmp_cpp
{
    DMPConfig::DMPConfig() : config_name("NOT_INITIALIZED!"), dmp_execution_time(0.0), fullyInitialized(false) 
    {}

    DMPConfig::DMPConfig(const string& filepath, const string& name) : config_name("NOT_INITIALIZED!"), dmp_execution_time(0.0), fullyInitialized(false)
    {
        if (!from_yaml_file(filepath, name))
        {
            stringstream ss;
            ss << "DMPModel: unable to load dmp config file: " << filepath;
            throw std::runtime_error(ss.str());
        }
    }

    bool DMPConfig::from_yaml_string(const string& yaml, const string& name)
    {
        stringstream sin(yaml);
        return from_yaml_istream(sin, name);
    }

    bool DMPConfig::from_yaml_file(const string& filepath, const string& name)
    {
        ifstream fin(filepath.c_str());
        return from_yaml_istream(fin, name);
    }

    bool DMPConfig::from_yaml_istream(istream& stream, string name)
    {
        string name_buf;
        vector<YAML::Node> all_docs = YAML::LoadAll(stream);
        for (size_t i = 0; i < all_docs.size(); i++)
        {
            YAML::Node doc = all_docs[i];
            if (doc["name"])
            {
                name_buf = doc["name"].as<std::string>();
                if (name == "")
                {
                    name = name_buf;
                }
                if (name_buf != name)
                {
                    continue;
                }
            }

            const bool has_been_initialized = fullyInitialized;
            bool new_config_is_complemete = true;
            config_name = name_buf;


            if (doc["dmp_execution_time"])
                dmp_execution_time = doc["dmp_execution_time"].as<double>();
            else 
                new_config_is_complemete = false;

            if(doc["dmp_startPos"])
                dmp_startPos = doc["dmp_startPos"].as<std::vector<double> >();
            else 
                new_config_is_complemete = false;

            if(doc["dmp_endPos"])
                dmp_endPos = doc["dmp_endPos"].as<std::vector<double> >();
            else 
                new_config_is_complemete = false;

            if(doc["dmp_startVel"])
                dmp_startVel = doc["dmp_startVel"].as<std::vector<double> >();
            else 
                new_config_is_complemete = false;

            if(doc["dmp_endVel"])
                dmp_endVel = doc["dmp_endVel"].as<std::vector<double> >();
            else 
                new_config_is_complemete = false;

            if(doc["dmp_startAcc"])
                dmp_startAcc = doc["dmp_startAcc"].as<std::vector<double> >();
            else 
                new_config_is_complemete = false;

            if(doc["dmp_endAcc"])
                dmp_endAcc = doc["dmp_endAcc"].as<std::vector<double> >();
            else 
                new_config_is_complemete = false;

            fullyInitialized = new_config_is_complemete || has_been_initialized;

            return is_valid();
        }
    }

    void DMPConfig::to_yaml_file(string filepath)
    {
        YAML::Emitter out;
        out << YAML::BeginDoc;
        out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << config_name;
        out << YAML::Key << "dmp_execution_time"     << YAML::Value << dmp_execution_time;
        out << YAML::Key << "dmp_startPos"           << YAML::Value << YAML::Flow << dmp_startPos;
        out << YAML::Key << "dmp_endPos"             << YAML::Value << YAML::Flow << dmp_endPos;
        out << YAML::Key << "dmp_startVel"           << YAML::Value << YAML::Flow << dmp_startVel;
        out << YAML::Key << "dmp_endVel"             << YAML::Value << YAML::Flow << dmp_endVel;
        out << YAML::Key << "dmp_startAcc"           << YAML::Value << YAML::Flow << dmp_startAcc;
        out << YAML::Key << "dmp_endAcc"             << YAML::Value << YAML::Flow << dmp_endAcc;
        out << YAML::EndMap;
        out << YAML::EndDoc;

        ofstream fout(filepath.c_str(), ios::out | ios::app);
        fout << out.c_str();
        fout.close();
    }

    bool DMPConfig::is_valid() const 
    {
        //cout << dmp_startPos.size() << " " << dmp_endPos.size() << " " << dmp_startVel.size() << " " << dmp_endVel.size() << " " << dmp_startAcc.size() << " " << dmp_endAcc.size() << std::endl;
        std::stringstream ss;
        bool valid = fullyInitialized;
        if (dmp_startPos.size() != dmp_endPos.size() ||
            dmp_startPos.size() != dmp_startVel.size() ||
            dmp_startPos.size() != dmp_endVel.size() ||
            dmp_startPos.size() != dmp_startAcc.size() ||
            dmp_startPos.size() != dmp_endAcc.size())
        {
            valid = false;

            ss << "DMPConfig not valid. All dimensions need to be equal.";
            throw std::runtime_error(ss.str());
        }

        if (dmp_execution_time <= 0.0)
        {
            valid = false;

            ss << "DMPConfig not valid. Execution time should be > 0.";
            throw std::runtime_error(ss.str());
            
        }
        
        if (valid)
        {
            for (size_t i = 0; i < dmp_startPos.size(); i++)
            {
                if (std::isnan(dmp_startPos[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_startPos contains NAN.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isnan(dmp_endPos[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_endPos contains NAN.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isnan(dmp_startVel[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_startVel contains NAN.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isnan(dmp_endVel[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_endVel contains NAN.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isnan(dmp_startAcc[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_startAcc contains NAN.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isnan(dmp_endAcc[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_endAcc contains NAN.";
                    throw std::runtime_error(ss.str());
                }

                 if (std::isinf(dmp_startPos[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_startPos contains INF.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isinf(dmp_endPos[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_endPos contains INF.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isinf(dmp_startVel[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_startVel contains INF.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isinf(dmp_endVel[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_endVel contains INF.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isinf(dmp_startAcc[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_startAcc contains INF.";
                    throw std::runtime_error(ss.str());
                }

                if (std::isinf(dmp_endAcc[i]))
                {
                    valid = false;

                    ss << "DMPConfig not valid. dmp_endAcc contains INF.";
                    throw std::runtime_error(ss.str());
                }
            }
        }

        return valid;
    }
}

