cmake_minimum_required(VERSION 2.8.11)

project(voly-labeller)

option(test "Build all tests." ON)
if(test)
  enable_testing()
  add_subdirectory(test)
endif(test)


include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(CMAKE_CXX_FLAGS_MEMORY "${CMAKE_CXX_FLAGS} -O1 -g -fsanitize=memory -fno-omit-frame-pointer" CACHE STRING
  "Flags used by the C++ compiler during memory sanitization builds." FORCE)

set(CMAKE_CXX_FLAGS_ADDRESS "${CMAKE_CXX_FLAGS} -O1 -g -fsanitize=address -fsanitize-blacklist=${CMAKE_SOURCE_DIR}/blacklist.txt -fno-omit-frame-pointer" CACHE STRING # -stdlib=libc++
  "Flags used by the C++ compiler during address sanitization builds." FORCE)

set(SOURCES
  src/window.cpp
  src/camera.cpp
  src/shader_program.cpp
  src/mesh.cpp
  src/main.cpp
  src/abstract_scene.h
  src/demo_scene.cpp
  src/input/scxml_importer.cpp
  ${RESOURCES})

add_executable(voly-labeller ${SOURCES})

target_link_libraries(voly-labeller
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Gui
  Qt5::Quick
  ${OPENGL_LIBRARIES}
  ${ASSIMP_LIBRARIES}
)

add_custom_target(valgrind valgrind -v --leak-check=full --show-leak-kinds=all --tool=memcheck --suppressions=../config/valgrind_suppressions.txt ./voly-labeller)
add_custom_target(helgrind valgrind -v --tool=helgrind ./voly-labeller)
# Analyzers
add_custom_target(lint python ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpplint.py --root=${CMAKE_CURRENT_SOURCE_DIR} --filter=-whitespace/braces,-legal/copyright,-readability/streams,-runtime/references,-runtime/int,-whitespace/indent ${CMAKE_CURRENT_SOURCE_DIR}/src/*.* ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.*)
add_custom_target(cppcheck cppcheck ${CMAKE_CURRENT_SOURCE_DIR}/src --enable=all VERBATIM)

# Documentation
add_custom_target(doc cd ${CMAKE_CURRENT_SOURCE_DIR} && doxygen)
add_custom_target(doc-open xdg-open ${CMAKE_CURRENT_SOURCE_DIR}/doc/html/index.html)

