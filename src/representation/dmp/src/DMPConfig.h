/*************************************************************************
	> File Name: DMPConfig.h
	> Author: Yi Zheng 
	> Mail: hczhengcq@gmail.com
	> Created Time: Thu 26 Apr 2018 02:12:16 AM PDT
 ************************************************************************/
#pragma once
#ifndef _DMPCONFIG_H
#define _DMPCONFIG_H

#include <string>
#include <vector>
#include <istream>

namespace dmp_cpp
{
    struct DMPConfig
    {
        std::string config_name;
        double dmp_execution_time; /**<Execution time of the dmp system, I.e. the time that it takes for the dmp system to get from the start position to the end position*/
        std::vector<double> dmp_startPos; /**<Start position of the trajectory*/
        std::vector<double> dmp_endPos; /**<End position of the trajectory*/
        std::vector<double> dmp_startVel; /**<Start velocity of the trajectory*/
        std::vector<double> dmp_endVel; /**<End velocity of the trajectory*/
        std::vector<double> dmp_startAcc; /**<Start acceleration of the trajectory */
        std::vector<double> dmp_endAcc; /**<End acceleration of the trajectory */

        DMPConfig();
        DMPConfig(const std::string& filepath, const std::string& name);
        
        bool from_yaml_file(const std::string& filepath, const std::string& name);
        bool from_yaml_string(const std::string&, const std::string& name);
        bool from_yaml_istream(std::istream& stream, std::string name);
        void to_yaml_file(std::string filepath);

        bool is_valid() const;

        // whether all attributes have been initialized by any of the from_yaml_methods
        bool fullyInitialized;
    };
}
#endif
