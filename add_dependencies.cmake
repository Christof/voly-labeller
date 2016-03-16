set(LIBRARIES)

find_package(Qt5Core 5.5 REQUIRED)
find_package(Qt5Gui 5.5 REQUIRED)
find_package(Qt5Widgets 5.5 REQUIRED)
find_package(Qt5OpenGL 5.5 REQUIRED)
find_package(Qt5OpenGLExtensions 5.5 REQUIRED)
find_package(Qt5Quick 5.5 REQUIRED)
find_package(Qt5Xml 5.5 REQUIRED)

find_package(CUDA 7.0 REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE_BUILD OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

#############################
# Check for GPUs present and their compute capability
# based on http://stackoverflow.com/questions/2285185/easiest-way-to-test-for-existence-of-cuda-capable-gpu-from-cmake/2297877#2297877 (Christopher Bruns)
if(CUDA_FOUND)

  set(CUDA_cuda_compute_capability ${CMAKE_SOURCE_DIR}/scripts/cuda_compute_capability.c)

  try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
    ${CMAKE_BINARY_DIR}
    ${CUDA_cuda_compute_capability}
    CMAKE_FLAGS
    -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
    -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
  # COMPILE_RESULT_VAR is TRUE when compile succeeds
  # RUN_RESULT_VAR is zero when a GPU is found
  if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
    set(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
    set(LIST_OF_GPU_ARCHS ${RUN_OUTPUT_VAR})
    list(FIND LIST_OF_GPU_ARCHS "21" found21)
    if(${found21} GREATER "-1")
      list(REMOVE_AT LIST_OF_GPU_ARCHS ${found21})
      list(APPEND LIST_OF_GPU_ARCHS "20")
    endif()
    list(REMOVE_DUPLICATES LIST_OF_GPU_ARCHS)
    set(CUDA_COMPUTE_CAPABILITY ${LIST_OF_GPU_ARCHS} CACHE STRING "Compute capability of CUDA-capable GPUs present")
    #set(CUDA_GENERATE_CODE "arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}" CACHE STRING "Which GPU architectures to generate code for (each arch/code pair will be passed as --generate-code option to nvcc, separate multiple pairs by ;)")
    mark_as_advanced(CUDA_COMPUTE_CAPABILITY)
  else()
    set(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
  endif()
  message(STATUS "initial cuda arch flags: |${CUDA_ARCH_FLAGS}|")
  set(CUDA_ARCH_FLAGS )
  foreach(code ${CUDA_COMPUTE_CAPABILITY})
    set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS} "-gencode;arch=compute_${code},code=sm_${code}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_${code},code=sm_${code}")
  endforeach()

  MESSAGE(STATUS "CUDA: automatically determined cuda nvcc flags: ${CUDA_NVCC_FLAGS}")
  MESSAGE(STATUS "CUDA: arch flags: ${CUDA_ARCH_FLAGS}")
endif(CUDA_FOUND)


if (${CUDA_VERSION} VERSION_GREATER "6")
  #message(status "Cuda version > 6 found: ${CUDA_VERSION}")
  message(status "CudaRT libraries ${CUDA_CUDART_LIBRARY} dir:${CUDA_TOOLKIT_ROOT_DIR}")
  find_library(CUDA_CUDARTDEV_LIBRARY cudadevrt PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib )
else()
  set(CUDA_CUDARTDEV_LIBRARY)
endif()
list(APPEND CUDA_LIBRARIES ${CUDA_CUDARTDEV_LIBRARY})

add_definitions(-DUSECUDA)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND LIBRARIES ${CUDA_LIBRARIES})


find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
list(APPEND LIBRARIES ${OPENGL_LIBRARIES})

find_package(PkgConfig)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})
list(APPEND LIBRARIES ${OPENGL_LIBRARIES})

pkg_check_modules(ASSIMP REQUIRED assimp)
include_directories(${ASSIMP_INCLUDE_DIRS})
list(APPEND LIBRARIES ${ASSIMP_LIBRARIES})

include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})
include_directories(${Qt5OpenGLExtensions_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})
include_directories(${Qt5Quick_INCLUDE_DIRS})
include_directories(${Qt5Xml_INCLUDE_DIRS})
list(APPEND LIBRARIES
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::OpenGLExtensions
  Qt5::Gui
  Qt5::Quick
  Qt5::Xml
)

set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost 1.58.0 COMPONENTS date_time timer filesystem system serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
list(APPEND LIBRARIES ${Boost_LIBRARIES})

find_package(ITK 4.5 REQUIRED)
include(${ITK_USE_FILE})
add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
add_definitions("-DVCL_NEEDS_INLINE_INSTANTIATION=0")
list(APPEND LIBRARIES ${ITK_LIBRARIES})

find_package(ImageMagick REQUIRED COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})
add_definitions("-DMAGICKCORE_QUANTUM_DEPTH=32")
add_definitions("-DMAGICKCORE_HDRI_ENABLE=1")
list(APPEND LIBRARIES ${ImageMagick_LIBRARIES})

#  -Wno-unused-local-typedefs is just because of ITK 4.5 with 4.7 it is not necessary any more
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -g -fPIC  -Wno-unused-local-typedefs")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTORCC ON) needs cmake 3.2.1
set(CMAKE_AUTOMOC ON)

qt5_add_resources(RESOURCES ${CMAKE_SOURCE_DIR}/src/resources.qrc)
