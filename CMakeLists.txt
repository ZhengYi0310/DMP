cmake_minimum_required(VERSION 2.6)


project(DMP)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Dynamical Movement Primitives.")


IF (WIN32)
  SET (POSTLIB ".dll")
  SET (PRELIB "lib")
  SET (LIBPATH "bin")
  SET (LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
ELSE (WIN32)
  IF (APPLE)
    SET (POSTLIB ".dylib")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
    SET (LIBRARY_PATH_VAR "DYLD_LIBRARY_PATH")
  ELSE (APPLE)
    SET (POSTLIB ".so")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
    SET (LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
  ENDIF (APPLE)
ENDIF (WIN32)

add_definitions ("-Wall")



find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)
pkg_check_modules(EIGEN3 REQUIRED eigen3)

add_subdirectory(implementation)
