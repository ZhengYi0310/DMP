
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${LIB_MANAGER_INCLUDE_DIRS}
    .
    )

link_directories(${YAMLCPP_LIBRARY_DIRS})

set(dmp_sources
    test_dmp_config.cpp
    test_dmp_model.cpp
    test_main.cpp)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model.yaml DESTINATION .)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml DESTINATION .)

add_executable (dmp_cpp_test ${dmp_sources})
target_link_libraries(dmp_cpp_test dmp_cpp ${LIB_MANAGER_LIBRARIES})

enable_testing()
add_test(dmp_test dmp_cpp_test)
