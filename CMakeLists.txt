cmake_minimum_required(VERSION 2.8.11)

project(voly-labeller)

add_subdirectory(src/math)
add_subdirectory(src/input)
add_subdirectory(src/utils)
add_subdirectory(src/labelling)
add_subdirectory(src/forces)

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

set(CMAKE_CXX_FLAGS_COV "${CMAKE_CXX_FLAGS} -O0 -g --coverage -fprofile-arcs -ftest-coverage" CACHE STRING
    "Flags used by the C++ compiler during coverage builds for GCC." FORCE)

set(SOURCES
  src/window.cpp
  src/gl.cpp
  src/eigen_qdebug.cpp
  src/camera.cpp
  src/shader_program.cpp
  src/render_object.cpp
  src/renderable.cpp
  src/mesh.cpp
  src/quad.cpp
  src/frame_buffer_object.cpp
  src/connector.cpp
  src/cube.cpp
  src/texture.cpp
  src/importer.cpp
  src/mesh_node.cpp
  src/label_node.cpp
  src/obb_node.cpp
  src/volume_node.cpp
  src/forces_visualizer_node.cpp
  src/coordinate_system_node.cpp
  src/nodes.cpp
  src/abstract_scene.h
  src/scene.cpp
  src/frustum_optimizer.cpp
  src/camera_controller.cpp
  src/mouse_dragging_controller.cpp
  src/camera_rotation_controller.cpp
  src/camera_zoom_controller.cpp
  src/camera_move_controller.cpp
  src/picking_controller.cpp
  src/mouse_shape_controller.h
  src/labeller_model.cpp
  src/labels_model.cpp
  src/volume_reader.cpp
  src/default_scene_creator.cpp
  ${RESOURCES})

add_library(voly-labeller-lib OBJECT ${SOURCES})

add_executable(voly-labeller
  src/main.cpp
  $<TARGET_OBJECTS:voly-labeller-lib>
  $<TARGET_OBJECTS:input>
  $<TARGET_OBJECTS:utils>
  $<TARGET_OBJECTS:labelling>
  $<TARGET_OBJECTS:forces>
  $<TARGET_OBJECTS:math>
  )

target_link_libraries(voly-labeller
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Gui
  Qt5::Quick
  ${OPENGL_LIBRARIES}
  ${ASSIMP_LIBRARIES}
  ${Boost_LIBRARIES}
  ${ITK_LIBRARIES}
)

add_custom_target(valgrind valgrind -v --leak-check=full --show-leak-kinds=all --tool=memcheck --suppressions=../config/valgrind_suppressions.txt ./voly-labeller)
add_custom_target(helgrind valgrind -v --tool=helgrind ./voly-labeller)
# Analyzers
add_custom_target(lint python ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpplint.py --root=${CMAKE_CURRENT_SOURCE_DIR} --filter=-whitespace/braces,-legal/copyright,-readability/streams,-runtime/references,-runtime/int,-whitespace/indent ${CMAKE_CURRENT_SOURCE_DIR}/src/*.* ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.*)
add_custom_target(cppcheck cppcheck ${CMAKE_CURRENT_SOURCE_DIR}/src --enable=all VERBATIM)

# Testing
add_custom_target(unit test/tests --gtest_filter=Test*)

# Coverage
set(LCOV_DIRS  -b ${CMAKE_CURRENT_SOURCE_DIR}/src --directory . )
add_custom_target(cov
  rm -f coverage.* &&
  lcov -z ${LCOV_DIRS} &&
  make unit &&
  lcov --compat split_crc=off --capture ${LCOV_DIRS} --output-file coverage.info &&
  lcov --remove coverage.info 'usr/*' 'boost/*' 'test/*' 'tmp/*' '**/moc_*' 'build*/*' --output-file coverage.info.cleaned &&
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/complete_coverage.sh &&
  genhtml coverage.info.complete --output-directory coverage-output --ignore-errors source)

add_custom_target(cov-open xdg-open ./coverage-output/index.html)

# Documentation
add_custom_target(doc cd ${CMAKE_CURRENT_SOURCE_DIR} && doxygen)
add_custom_target(doc-open xdg-open ${CMAKE_CURRENT_SOURCE_DIR}/doc/html/index.html)

