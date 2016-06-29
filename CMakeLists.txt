cmake_minimum_required(VERSION 2.8.11)

project(voly-labeller)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

add_subdirectory(src/math)
add_subdirectory(src/input)
add_subdirectory(src/utils)
add_subdirectory(src/labelling)
add_subdirectory(src/forces)
add_subdirectory(src/placement)

if (WIN32)
  include_directories(${PROJECT_SOURCE_DIR}/msincludes)
endif()
add_subdirectory(external/VolyVideoModule)

option(test "Build all tests." ON)
if(test)
  enable_testing()
  add_subdirectory(test)
endif(test)


include("${CMAKE_SOURCE_DIR}/add_dependencies.cmake")

set(CLANG_FLAGS "-Wno-unknown-warning-option -Wno-inconsistent-missing-override")

set(CMAKE_CXX_FLAGS_MEMORY "${CMAKE_CXX_FLAGS} -O1 -g -fsanitize=memory -fno-omit-frame-pointer ${CLANG_FLAGS}" CACHE STRING
  "Flags used by the C++ compiler during memory sanitization builds." FORCE)

set(CMAKE_CXX_FLAGS_ADDRESS "${CMAKE_CXX_FLAGS} -O1 -g -fsanitize=address -fsanitize-blacklist=${CMAKE_SOURCE_DIR}/blacklist.txt -fno-omit-frame-pointer ${CLANG_FLAGS}" CACHE STRING # -stdlib=libc++
  "Flags used by the C++ compiler during address sanitization builds." FORCE)

set(CMAKE_CXX_FLAGS_COV "${CMAKE_CXX_FLAGS} -O0 -g --coverage -fprofile-arcs -ftest-coverage -Wno-deprecated-declarations" CACHE STRING
  "Flags used by the C++ compiler during coverage builds for GCC." FORCE)

set(SOURCES
  src/application.cpp
  src/window.cpp
  src/eigen_qdebug.cpp
  src/camera.cpp
  src/importer.cpp
  src/mesh_node.cpp
  src/label_node.cpp
  src/obb_node.cpp
  src/volume_node.cpp
  src/forces_visualizer_node.cpp
  src/coordinate_system_node.cpp
  src/camera_node.cpp
  src/nodes.cpp
  src/nodes_controller.cpp
  src/abstract_scene.h
  src/scene.cpp
  src/scene_controller.cpp
  src/texture_mapper_manager.cpp
  src/texture_mappers_for_layer.cpp
  src/texture_mapper_manager_controller.cpp
  src/frustum_optimizer.cpp
  src/camera_controller.cpp
  src/mouse_dragging_controller.cpp
  src/camera_rotation_controller.cpp
  src/camera_zoom_controller.cpp
  src/camera_move_controller.cpp
  src/camera_controllers.cpp
  src/picking_controller.cpp
  src/picker.cpp
  src/mouse_shape_controller.h
  src/labeller_model.cpp
  src/placement_labeller_model.cpp
  src/labels_model.cpp
  src/camera_positions_model.cpp
  src/labelling_coordinator.cpp
  src/labelling_controller.cpp
  src/volume_reader.cpp
  src/default_scene_creator.cpp
  src/constraint_buffer_object.cpp
  src/video_recorder.cpp
  src/video_recorder_controller.cpp
  src/graphics/gl.cpp
  src/graphics/shader_program.cpp
  src/graphics/buffer.cpp
  src/graphics/render_data.cpp
  src/graphics/renderable.cpp
  src/graphics/quad.cpp
  src/graphics/screen_quad.cpp
  src/graphics/connector.cpp
  src/graphics/cube.cpp
  src/graphics/mesh.cpp
  src/graphics/frame_buffer_object.cpp
  src/graphics/ha_buffer.cpp
  src/graphics/timer.cpp
  src/graphics/buffer_hole_manager.cpp
  src/graphics/attribute_buffer.cpp
  src/graphics/buffer_lock_manager.cpp
  src/graphics/shader_buffer.cpp
  src/graphics/mapped_buffer.h
  src/graphics/circular_buffer.h
  src/graphics/buffer_manager.cpp
  src/graphics/texture_address.h
  src/graphics/texture_container.cpp
  src/graphics/texture2d.cpp
  src/graphics/texture_manager.cpp
  src/graphics/texture_space_description.cpp
  src/graphics/object_manager.cpp
  src/graphics/object_data.cpp
  src/graphics/shader_manager.cpp
  src/graphics/transfer_function_manager.cpp
  src/graphics/volume.h
  src/graphics/volume_data.h
  src/graphics/volume_manager.cpp
  src/graphics/managers.cpp
  src/graphics/standard_texture_2d.cpp
  src/graphics/vertex_buffer.h
  src/graphics/vertex_array.cpp
  src/graphics/drawer.h
  src/graphics/buffer_drawer.cpp
  src/graphics/qimage_drawer.cpp
  src/graphics/random_texture_generator.cpp
  src/ui.qml
  src/ForcesTableView.qml
  src/LabelsTableView.qml
  src/PlacementWeightsTableView.qml
  src/CameraPositionsView.qml
  src/NumberTextSliderInput.qml
  ${RESOURCES})

file(GLOB SHADER src/shader/*)

add_library(voly-labeller-lib OBJECT ${SOURCES})

set(ALL_TARGET_OBJECTS)
set(ALL_TARGET_LIBRARIES)

if (MSVC)
  set(ALL_TARGET_OBJECTS
    ${ALL_TARGET_OBJECTS}
    $<TARGET_OBJECTS:voly-labeller-lib>
    )
  set(ALL_TARGET_LIBRARIES
    ${ALL_TARGET_LIBRARIES}
    #voly-labeller-lib
    input
    utils
    labelling
    forces
    math
    )
else()
  set(ALL_TARGET_OBJECTS
    ${ALL_TARGET_OBJECTS}
    $<TARGET_OBJECTS:voly-labeller-lib>
    $<TARGET_OBJECTS:input>
    $<TARGET_OBJECTS:utils>
    $<TARGET_OBJECTS:labelling>
    $<TARGET_OBJECTS:forces>
    $<TARGET_OBJECTS:math>
    )
endif()

cuda_add_executable(voly-labeller
  src/main.cpp
  ${SHADER}
  ${ALL_TARGET_OBJECTS}
  )

target_link_libraries(voly-labeller ${LIBRARIES} ${ALL_TARGET_LIBRARIES} placement-lib ${CUDA_CUDARTDEV_LIBRARY} volyvideomodule)

add_custom_target(valgrind valgrind -v --leak-check=full --show-leak-kinds=all --tool=memcheck --suppressions=../config/valgrind_suppressions.txt ./voly-labeller)
add_custom_target(helgrind valgrind -v --tool=helgrind ./voly-labeller)
# Analyzers
add_custom_target(lint python ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cpplint.py --root=${CMAKE_CURRENT_SOURCE_DIR}
  --filter=-whitespace/braces,-legal/copyright,-readability/streams,-runtime/references,-runtime/int,-whitespace/indent,-build/c++11
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.* ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.*)
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

# Profiling
set(executable "./${PROJECT_NAME}")
set(profile profile_${PROJECT_NAME})
add_custom_target(profile_${PROJECT_NAME}
  COMMAND LD_PRELOAD=/usr/local/lib/libprofiler.so CPUPROFILE=${profile} ${executable} \${PARAMS} && pprof ${executable} ${profile} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_custom_target(callgrind_${PROJECT_NAME}
  COMMAND pprof --callgrind ${executable} ${profile} > ${profile}.callgrind && kcachegrind ${profile}.callgrind WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Installing - Visual Studio
if (MSVC)
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
  include(InstallRequiredSystemLibraries)
  install(FILES
    ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
    DESTINATION "."
    COMPONENT Runtime
    )

  GET_FILENAME_COMPONENT(CUDA_DIR [HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA\ Corporation\\GPU\ Computing\ Toolkit\\CUDA\\v7.5;InstallDir] REALPATH CACHE)
  if(VOLY_WIN32)
    set(machine_type "x86")
  elseif(VOLY_WIN64)
    set(machine_type "x64")
  endif()

  if(NOT EXISTS ${CUDA_DIR})
    message(" CUDA Toolkit directory not found: ${CUDA_DIR}")
  else()
    MESSAGE("  - ${CUDA_DIR}/bin/cudart64_75.dll")
    INSTALL(FILES "${CUDA_DIR}/bin/cudart64_75.dll"
      DESTINATION "."
      COMPONENT Runtime
      )
  endif()
  get_target_property(QtCore_location Qt5::Core LOCATION)
  get_target_property(QtWidgets_location Qt5::Widgets LOCATION)
  get_target_property(QtOpenGL_location Qt5::OpenGL LOCATION)
  get_target_property(QtGui_location Qt5::Gui LOCATION)
  get_target_property(QtXml_location Qt5::Xml LOCATION)
  get_target_property(QtNetwork_location Qt5::Network LOCATION)
  get_target_property(QtQml_location Qt5::Qml LOCATION)
  get_target_property(QtQuick_location Qt5::Quick LOCATION)

  get_target_property(QtCore_locationd Qt5::Core LOCATION_DEBUG)
  get_target_property(QtWidgets_locationd Qt5::Widgets LOCATION_DEBUG)
  get_target_property(QtOpenGL_locationd Qt5::OpenGL LOCATION_DEBUG)
  get_target_property(QtGui_locationd Qt5::Gui LOCATION_DEBUG)
  get_target_property(QtXml_locationd Qt5::Xml LOCATION_DEBUG)
  get_target_property(QtNetwork_locationd Qt5::Network LOCATION_DEBUG)
  get_target_property(QtQml_locationd Qt5::Qml LOCATION_DEBUG)
  get_target_property(QtQuick_locationd Qt5::Quick LOCATION_DEBUG)

  get_filename_component(Qt5_DLL_Path ${QtCore_location} PATH)
  get_filename_component(Qt5_plugin_Path "${Qt5_DLL_Path}/../plugins/platforms/" ABSOLUTE PATH)
  get_filename_component(Qt5_imageformats_Path "${Qt5_DLL_Path}/../plugins/imageformats/" ABSOLUTE PATH)
  message("Qt5 plugin path: ${Qt5_plugin_Path}")

  set(Qt5_Dependency_DLLS
    "${Qt5_DLL_Path}/icuin54.dll"
    "${Qt5_DLL_Path}/icuuc54.dll"
    "${Qt5_DLL_Path}/icudt54.dll"
    )
  set(Qt5_Platforms_Dependency_DLLS
    "${Qt5_plugin_Path}/qwindows.dll"
    "${Qt5_plugin_Path}/qwindowsd.dll"
    )
  set(Qt5_Imageformats_Dependency_DLLS
    "${Qt5_imageformats_Path}/qgif.dll"
    "${Qt5_imageformats_Path}/qico.dll"
    "${Qt5_imageformats_Path}/qjpeg.dll"
    "${Qt5_imageformats_Path}/qtiff.dll"
    )
  install(TARGETS voly-labeller
    RUNTIME DESTINATION "."
    )

  install(FILES
    ${QtCore_location} ${QtCore_locationd}
    ${QtWidgets_location} ${QtWidgets_locationd}
    ${QtOpenGL_location} ${QtOpenGL_locationd}
    ${QtGui_location} ${QtGui_locationd}
    ${QtXml_location} ${QtXml_locationd}
    ${QtNetwork_location} ${QtNetwork_locationd}
    ${QtQml_location} ${QtQml_locationd}
    ${QtQuick_location} ${QtQuick_locationd}
    ${Qt5_Dependency_DLLS}
    DESTINATION "."
    )
  install(FILES
    ${Qt5_Platforms_Dependency_DLLS}
    DESTINATION "platforms"
    )

  install(FILES
    ${Qt5_Imageformats_Dependency_DLLS}
    DESTINATION "imageformats"
    )

  message(STATUS "lib dir assimp: ${ASSIMP_LIBRARIES_DIR}")
  install(DIRECTORY ${ASSIMP_LIBRARIES_DIR}/
    DESTINATION "."
    FILES_MATCHING
    PATTERN "*.dll"
    )

  #install(FILES ${Clipper_INCLUDE_DIRS}/../bin/polyclipping.dll
  #  DESTINATION "."
  #  )

  install(DIRECTORY ${BOOST_LIBRARY_DIR_RELEASE}
    DESTINATION "."
    FILES_MATCHING
    PATTERN "*.dll"
    )

  install(DIRECTORY ${ITK_DIR}/../../../bin/
    DESTINATION "."
    FILES_MATCHING
    PATTERN "*.dll"
    )

endif()
