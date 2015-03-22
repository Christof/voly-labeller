cmake_minimum_required(VERSION 2.8.11)

project(tests)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(Threads REQUIRED)

file(GLOB_RECURSE TESTS ${CMAKE_CURRENT_SOURCE_DIR}**/*.cc)
add_executable(tests ${TESTS})
target_link_libraries(tests ${GTEST_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_test(tests tests)
