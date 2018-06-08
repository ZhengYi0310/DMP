cmake_minimum_required(VERSION 2.8)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#find_path(EIGEN3_INCLUDE_DIRS Eigen/Dense
#  ${CMAKE_INSTALL_PREFIX}/include/eigen3
#  /usr/include/eigen3
#  /opt/local/include/eigen3
#  DOC "Eigen 3 include directory")

#set(EIGEN3_FOUND ${EIGEN3_INCLUDE_DIRS} CACHE BOOL "" FORCE)
find_package(Eigen3)

enable_testing()

add_subdirectory(src)
add_subdirectory(dmp)
