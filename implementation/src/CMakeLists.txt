
find_path(EIGEN3_INCLUDE_DIRS Eigen/Dense
  ${CMAKE_INSTALL_PREFIX}/include/eigen3
  /usr/include/eigen3
  /opt/local/include/eigen3
  DOC "Eigen 3 include directory")

set(EIGEN3_FOUND ${EIGEN3_INCLUDE_DIRS} CACHE BOOL "" FORCE)
find_package(Eigen3)

include_directories(${EIGEN3_INCLUDE_DIRS} .)
add_library(dmp SHARED DMP.cpp)


install (TARGETS dmp DESTINATION lib)
install (FILES Dmp.h DESTINATION include/${PROJECT_NAME})
