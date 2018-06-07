cmake_minimum_required(VERSION 2.6)


project(DMP)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Dynamical Movement Primitives.")

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${LIB_MANAGER_INCLUDE_DIRS}
    include
)

link_directories(${YAMLCPP_LIBRARY_DIRS})
add_definitions(${YAMLCPP_CFLAGS_OTHER})  #flags excluding the ones with -I


set(sources src/DMPConfig.cpp
            src/DMPModel.cpp)
set(headers include/DMP/DMPConfig.h
            include/DMP/DMPModel.h)

add_library(dmp_cpp SHARED ${sources} ${headers})
target_link_libraries(dmp_cpp ${YAMLCPP_LIBRARIES})
#BUILD_DMP_BEHAVIOR_STANDALONE enables the behavior macros for the DmpBehavior
set_target_properties(dmp_cpp PROPERTIES COMPILE_FLAGS "-DBUILD_DMP_BEHAVIOR_STANDALONE")

install (TARGETS dmp_cpp DESTINATION lib)
install (FILES ${headers} DESTINATION include/${PROJECT_NAME})

enable_testing()
add_subdirectory(test)
