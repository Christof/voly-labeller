cmake_minimum_required(VERSION 2.8.11)

# including itk 4.8.1 before building cuda-dependend libraries produces a cmake error 
# because itk defines some invalid nvcc-flags in the generated cmake files
find_package(ITK 4.5 REQUIRED)
include(${ITK_USE_FILE})
add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
add_definitions("-DVCL_NEEDS_INLINE_INSTANTIATION=0")

if(${ITK_VERSION_MAJOR}==4 AND ${ITK_VERSION_MINOR}<7)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
endif()
