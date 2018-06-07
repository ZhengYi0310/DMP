/*************************************************************************
	> File Name: test_dmp_config.cpp
	> Author: 
	> Mail: 
	> Created Time: Thu 26 Apr 2018 05:16:14 AM PDT
 ************************************************************************/

#include <catch/catch.hpp>
#include <DMP/DMPConfig.h>
#include "test_helpers.h"
#include <fstream>
#include <string>
#include <stdio.h>  /* defines FILENAME_MAX */
#include <unistd.h>

using namespace dmp_cpp;

bool have_config=false;
DMPConfig initial_config;
bool have_other_config=false;
DMPConfig other_config;

void make_config(){
    if(have_config)
        return;
    initial_config.config_name = "a name";
    initial_config.dmp_execution_time = TestHelpers::random_double();
    initial_config.dmp_startPos = TestHelpers::random_vector(5);
    initial_config.dmp_endPos = TestHelpers::random_vector(5);
    initial_config.dmp_startVel = TestHelpers::random_vector(5);
    initial_config.dmp_endVel = TestHelpers::random_vector(5);
    initial_config.dmp_startAcc = TestHelpers::random_vector(5);
    initial_config.dmp_endAcc = TestHelpers::random_vector(5);
    have_config = true;
}

void make_other_config(){
    if(have_other_config)
        return;
    other_config.config_name = "another name";
    other_config.dmp_execution_time = TestHelpers::random_double();
    other_config.dmp_startPos = TestHelpers::random_vector(5);
    other_config.dmp_endPos = TestHelpers::random_vector(5);
    other_config.dmp_startVel = TestHelpers::random_vector(5);
    other_config.dmp_endVel = TestHelpers::random_vector(5);
    other_config.dmp_startAcc = TestHelpers::random_vector(5);
    other_config.dmp_endAcc = TestHelpers::random_vector(5);
    have_other_config = true;
}

void compare_configs(DMPConfig config_a, DMPConfig config_b)
{

  REQUIRE(config_a.dmp_startPos.size() == config_a.dmp_endPos.size());
  REQUIRE(config_a.dmp_startPos.size() == config_a.dmp_startVel.size());
  REQUIRE(config_a.dmp_startPos.size() == config_a.dmp_endVel.size());
  REQUIRE(config_a.dmp_startPos.size() == config_a.dmp_startAcc.size());
  REQUIRE(config_a.dmp_startPos.size() == config_a.dmp_endAcc.size());

  REQUIRE(config_a.dmp_startPos.size() == config_b.dmp_startPos.size());
  REQUIRE(config_a.dmp_endPos.size() == config_b.dmp_endPos.size());
  REQUIRE(config_a.dmp_startVel.size() == config_b.dmp_startVel.size());
  REQUIRE(config_a.dmp_endVel.size() == config_b.dmp_endVel.size());
  REQUIRE(config_a.dmp_startAcc.size() == config_b.dmp_startAcc.size());
  REQUIRE(config_a.dmp_endAcc.size() == config_b.dmp_endAcc.size());

  REQUIRE(config_a.dmp_execution_time == Approx(config_b.dmp_execution_time));

  for(unsigned i = 0; i < config_a.dmp_startPos.size(); ++i)
  {
    REQUIRE(config_a.dmp_startPos[i] == Approx(config_b.dmp_startPos[i]));
    REQUIRE(config_a.dmp_endPos[i] == Approx(config_b.dmp_endPos[i]));
    REQUIRE(config_a.dmp_startVel[i] == Approx(config_b.dmp_startVel[i]));
    REQUIRE(config_a.dmp_endVel[i] == Approx(config_b.dmp_endVel[i]));
    REQUIRE(config_a.dmp_startAcc[i] == Approx(config_b.dmp_startAcc[i]));
    REQUIRE(config_a.dmp_endAcc[i] == Approx(config_b.dmp_endAcc[i]));
  }
}

TEST_CASE("Test DMPConfig", "[DMPConfig]") {
    make_config();
    make_other_config();

    char cp[1000];
    getcwd(cp, sizeof(cp));
    std::string current_path = cp;
    std::string filepath = current_path+"/import_export_test_config.yml";
    //std::cout << filepath;

    SECTION("Could Export to yaml") {
        //First check if test file is already there, delete if so
        TestHelpers::delete_file_if_exists(filepath);

        //Export
        REQUIRE_NOTHROW(initial_config.to_yaml_file(filepath));

        //Does file exist?
        std::ifstream ifile;
        ifile.open(filepath.c_str());
        REQUIRE(ifile.is_open());
        ifile.close();
    }

    SECTION("Could import") {
        //std::cout << filepath << std::endl;
        DMPConfig loaded_config;
        REQUIRE(loaded_config.from_yaml_file(filepath, initial_config.config_name));

        compare_configs(loaded_config, initial_config);
    }

    SECTION("Could append to file"){
        DMPConfig loaded_config;

        REQUIRE_NOTHROW(other_config.to_yaml_file(filepath));
        REQUIRE(loaded_config.from_yaml_file(filepath, initial_config.config_name));
        REQUIRE(loaded_config.config_name == initial_config.config_name);

        REQUIRE(loaded_config.from_yaml_file(filepath, other_config.config_name));
        REQUIRE(loaded_config.config_name == other_config.config_name);
    }

    SECTION("load from string") {
      DMPConfig config;
      const std::string yaml("name: ''\n"
                             "dmp_execution_time: 10\n"
                             "dmp_startPos: [0, 1]\n"
                             "dmp_endPos: [10, 11]\n"
                             "dmp_startVel: [1, 2]\n"
                             "dmp_endVel: [3, 4]\n"
                             "dmp_startAcc: [5, 6]\n"
                             "dmp_endAcc: [7, 8]");
      REQUIRE(config.from_yaml_string(yaml,""));
      REQUIRE(config.dmp_execution_time == 10.0);
      REQUIRE(config.dmp_startPos[0] == 0.0);
      REQUIRE(config.dmp_startPos[1] == 1.0);
      REQUIRE(config.dmp_endPos[0] == 10.0);
      REQUIRE(config.dmp_endPos[1] == 11.0);
      REQUIRE(config.dmp_startVel[0] == 1.0);
      REQUIRE(config.dmp_startVel[1] == 2.0);
      REQUIRE(config.dmp_endVel[0] == 3.0);
      REQUIRE(config.dmp_endVel[1] == 4.0);
      REQUIRE(config.dmp_startAcc[0] == 5.0);
      REQUIRE(config.dmp_startAcc[1] == 6.0);
      REQUIRE(config.dmp_endAcc[0] == 7.0);
      REQUIRE(config.dmp_endAcc[1] == 8.0);
    }
}
