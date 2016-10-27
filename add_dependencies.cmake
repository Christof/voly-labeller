set(LIBRARIES)

if( MSVC )
  # in order to prevent DLL hell, each of the DLLs have to be suffixed with the major version and msvc prefix
  if( MSVC70 OR MSVC71 )
    set(MSVC_PREFIX "vc70")
  elseif( MSVC80 )
    set(MSVC_PREFIX "vc80")
  elseif( MSVC90 )
    set(MSVC_PREFIX "vc90")
  elseif( MSVC10 )
    set(MSVC_PREFIX "vc100")
  elseif( MSVC11 )
    set(MSVC_PREFIX "vc110")
  elseif( MSVC12 )
    set(MSVC_PREFIX "vc120")
  elseif( MSVC14 )
    set(MSVC_PREFIX "vc140")
  else()
    set(MSVC_PREFIX "vc130")
  endif()
endif()


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
if(UNIX)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11)
else()
  if(MSVC)
    add_compile_options(/FS)
  endif()
endif()
if(MSVC)
  SET(CUDA_PROPAGATE_HOST_FLAGS ON)
else()
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

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
    if (${CUDA_VERSION} VERSION_EQUAL "7.5")
      message(STATUS "CUDA VERSION ${CUDA_VERSION}")
      if (${code} VERSION_GREATER "5.2")
        message(STATUS "Compute capability version ${code} too high, reset to 30")
        set(code 30)
      endif()
    endif()
    set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS} "-gencode;arch=compute_${code},code=compute_${code}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode;arch=compute_${code},code=compute_${code}")
  endforeach()

  MESSAGE(STATUS "CUDA: automatically determined cuda nvcc flags: ${CUDA_NVCC_FLAGS}")
  MESSAGE(STATUS "CUDA: arch flags: ${CUDA_ARCH_FLAGS}")
endif(CUDA_FOUND)

if(UNIX)
  if (${CUDA_VERSION} VERSION_GREATER "6")
    #message(STATUS "Cuda version > 6 found: ${CUDA_VERSION}")
    #message(STATUS "CudaRT libraries ${CUDA_CUDART_LIBRARY} dir:${CUDA_TOOLKIT_ROOT_DIR}")
    find_library(CUDA_CUDARTDEV_LIBRARY cudadevrt PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)
  else()
    set(CUDA_CUDARTDEV_LIBRARY)
  endif()
  list(APPEND CUDA_LIBRARIES ${CUDA_CUDARTDEV_LIBRARY})
endif()

add_definitions(-DUSECUDA)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND LIBRARIES ${CUDA_LIBRARIES})

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
list(APPEND LIBRARIES ${OPENGL_LIBRARIES})

if(UNIX)
  find_package(PkgConfig)
  pkg_check_modules(EIGEN3 REQUIRED eigen3)
  include_directories(${EIGEN3_INCLUDE_DIRS})
else()
  set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN_INCLUDE_DIR}" )
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Please point the environment variable EIGEN_INCLUDE_DIR to the include directory of your Eigen installation.")
  else()
    message(STATUS "Eigen3 include files found in: ${EIGEN3_INCLUDE_DIR}")
  endif()
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()


list(APPEND LIBRARIES ${OPENGL_LIBRARIES})


if(UNIX)
  pkg_check_modules(ASSIMP REQUIRED assimp)
  include_directories(${ASSIMP_INCLUDE_DIRS})
  list(APPEND LIBRARIES ${ASSIMP_LIBRARIES})
else()
  set(ASSIMP_DIR $ENV{ASSIMP_ROOT} $ENV{ASSIMP_DIR})
  #message(STATUS "ASSIMP_DIR: ${ASSIMP_DIR}")
  FIND_PATH(
    assimp_INCLUDE_DIRS
    NAMES assimp/postprocess.h assimp/scene.h assimp/version.h assimp/config.h assimp/cimport.h
    PATHS ${ASSIMP_DIR}
    PATH_SUFFIXES include
    )

  FIND_LIBRARY(
    assimp_LIBRARIES_DEBUG
    NAMES assimp assimp-${MSVC_PREFIX}-mtd
    PATHS ${ASSIMP_DIR}
    PATH_SUFFIXES lib lib64 lib32)

  FIND_LIBRARY(
    assimp_LIBRARIES_RELEASE
    NAMES assimp assimp-${MSVC_PREFIX}-mt
    PATHS ${ASSIMP_DIR}
    PATH_SUFFIXES lib lib64 lib32)

  FIND_FILE(assimp_DLL_DEBUG
    NAMES assimp.dll assimp-${MSVC_PREFIX}-mtd.dll
    PATHS ${ASSIMP_DIR}
    PATH_SUFFIXES bin
    )
  FIND_FILE(assimp_DLL_RELEASE
    NAMES assimp.dll assimp-${MSVC_PREFIX}-mt.dll
    PATHS ${ASSIMP_DIR}
    PATH_SUFFIXES bin
    )

  IF (assimp_INCLUDE_DIRS AND assimp_LIBRARIES_DEBUG AND assimp_LIBRARIES_RELEASE)
    SET(assimp_FOUND TRUE)
  ENDIF ()

  IF (assimp_FOUND)
    IF (NOT assimp_FIND_QUIETLY)
      MESSAGE(STATUS "Found asset importer library: ${assimp_LIBRARIES}")
    ENDIF (NOT assimp_FIND_QUIETLY)
    set(ASSIMP_INCLUDE_DIRS ${assimp_INCLUDE_DIRS})
    set(ASSIMP_LIBRARIES_DEBUG ${assimp_LIBRARIES_DEBUG})
    set(ASSIMP_LIBRARIES_RELEASE ${assimp_LIBRARIES_RELEASE})
    set(ASSIMP_LIBRARIES ${assimp_LIBRARIES_RELEASE})
    get_filename_component(ASSIMP_LIBRARIES_DIR ${assimp_DLL_RELEASE} DIRECTORY)

  ELSE (assimp_FOUND)
    MESSAGE(FATAL_ERROR "Could not find asset importer library")
  ENDIF (assimp_FOUND)
endif()

message(STATUS "Assimp include DIR: ${ASSIMP_INCLUDE_DIRS}")
message(STATUS "Assimp library DIR: ${ASSIMP_LIBRARIES_DIR}")
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
if(MSVC)
else()
  if(CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
      add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
    endif()
  endif()

  add_definitions("-DVCL_NEEDS_INLINE_INSTANTIATION=0")
  add_definitions("-D_MWAITXINTRIN_H_INCLUDED")
  add_definitions("-D_FORCE_INLINES")
endif()
list(APPEND LIBRARIES ${ITK_LIBRARIES})

find_package(ImageMagick REQUIRED COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})
if(UNIX)
  # are not added by find_package
  add_definitions("-DMAGICKCORE_QUANTUM_DEPTH=32")
  add_definitions("-DMAGICKCORE_HDRI_ENABLE=1")
endif()
list(APPEND LIBRARIES ${ImageMagick_LIBRARIES})

if(MSVC)
  try_run(CPU_TEST_RUN_RESULT_VAR CPU_TEST_COMPILE_RESULT_VAR
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/scripts/cpu_test.cpp
    RUN_OUTPUT_VARIABLE CPU_TEST_OUTPUT
    )
  #message(STATUS "CPU_TEST_RUN_RESULT_VAR: ${CPU_TEST_RUN_RESULT_VAR}")
  #message(STATUS "CPU_TEST_OUTPUT: ${CPU_TEST_OUTPUT}")

  list(FIND CPU_TEST_OUTPUT "SSE2" HAS_SSE2)
  list(FIND CPU_TEST_OUTPUT "USE_AVX" HAS_AVX)
  if (${HAS_SSE2} GREATER -1)
    set(HAS_SSE2 true)
  else()
    set(HAS_SSE2 false)
  endif()
  if (${HAS_AVX} GREATER -1)
    set(HAS_AVX true)
  else()
    set(HAS_AVX false)
  endif()

  #message(STATUS "SSE2: ${HAS_SSE2}   USE_AVX: ${HAS_AVX}")
  if (HAS_AVX)
    message(STATUS "Has AVX")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:AVX2 /FS /MP /openmp")
  else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /FS /MP /openmp")
  endif()
  add_definitions("-D_USE_MATH_DEFINES")
else()
  #  -Wno-unused-local-typedefs is just because of ITK 4.5 with 4.7 it is not necessary any more
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -g -fPIC  -Wno-unused-local-typedefs")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTORCC ON) needs cmake 3.2.1
set(CMAKE_AUTOMOC ON)

qt5_add_resources(RESOURCES ${CMAKE_SOURCE_DIR}/src/resources.qrc)
