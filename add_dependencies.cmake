
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5OpenGLExtensions REQUIRED)
find_package(Qt5Quick REQUIRED)

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
  message(ERROR "OPENGL not found!")
endif(NOT OPENGL_FOUND)

find_package(PkgConfig)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})
#find_package(Eigen3 REQUIRED)

pkg_check_modules(ASSIMP REQUIRED assimp)
include_directories(${ASSIMP_INCLUDE_DIRS})

include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})
include_directories(${Qt5OpenGLExtensions_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})
include_directories(${Qt5Quick_INCLUDE_DIRS})

find_package(Boost 1.57.0 COMPONENTS date_time filesystem system serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

FIND_PACKAGE (ITK 4.5 REQUIRED)
include(${ITK_USE_FILE})
add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
add_definitions("-DVCL_NEEDS_INLINE_INSTANTIATION=0")

#  -Wno-unused-local-typedefs is just because of ITK 4.5 with 4.7 it is not necessary any more
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -g -fPIC  -Wno-unused-local-typedefs")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTORCC ON) needs cmake 3.2.1
set(CMAKE_AUTOMOC ON)

qt5_add_resources(RESOURCES ${CMAKE_SOURCE_DIR}/src/resources.qrc)
