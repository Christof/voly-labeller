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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11;-gencode arch=compute_30,code=sm_30)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_definitions(-DUSECUDA)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND LIBRARIES ${CUDA_LIB})
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

find_package(Boost 1.59.0 COMPONENTS date_time timer filesystem system serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
list(APPEND LIBRARIES ${Boost_LIBRARIES})

FIND_PACKAGE (ITK 4.5 REQUIRED)
include(${ITK_USE_FILE})
add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
add_definitions("-DVCL_NEEDS_INLINE_INSTANTIATION=0")
list(APPEND LIBRARIES ${ITK_LIBRARIES})

find_package(ImageMagick REQUIRED COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})
list(APPEND LIBRARIES ${ImageMagick_LIBRARIES})

#  -Wno-unused-local-typedefs is just because of ITK 4.5 with 4.7 it is not necessary any more
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -g -fPIC  -Wno-unused-local-typedefs")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTORCC ON) needs cmake 3.2.1
set(CMAKE_AUTOMOC ON)

qt5_add_resources(RESOURCES ${CMAKE_SOURCE_DIR}/src/resources.qrc)
