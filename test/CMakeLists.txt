cmake_minimum_required(VERSION 2.8.11)

project(tests)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

qt5_add_resources(RESOURCES ../src/resources.qrc)

file(GLOB_RECURSE TESTS ${CMAKE_CURRENT_SOURCE_DIR}**/*.c*)
cuda_add_executable(tests ${TESTS} ${SOURCES}
  $<TARGET_OBJECTS:voly-labeller-lib>
  $<TARGET_OBJECTS:input>
  $<TARGET_OBJECTS:utils>
  $<TARGET_OBJECTS:labelling>
  $<TARGET_OBJECTS:forces>
  $<TARGET_OBJECTS:math>
  )

target_link_libraries(tests
  ${GTEST_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBRARIES}
  placement-lib
  volyvideomodule
  -lpthread
  )

add_test(tests tests)
