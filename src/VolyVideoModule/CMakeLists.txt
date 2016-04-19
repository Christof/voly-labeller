project(VolyVideoModule)

# This can not be moved into another cmake file!
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)

# Here we specify some basic settings, like cmake policies, cmake path extension from env vars and win32 specific settings
# INCLUDE(${CMAKE_SOURCE_DIR}/cmake/BasicCMakeSetup.cmake NO_POLICY_SCOPE)

find_package(Qt5Core REQUIRED)

set(FFMPEG_LIBRARIES )

find_library(FFMPEG_avcodec_LIBRARY NAMES avcodec-52 avcodec aviplayavcodec
  PATHS
  $ENV{FFMPEG_DIR}/lib
  /usr/lib
  /usr/local/lib
  /usr/lib64/lib
  /usr/local/lib64
)
if (FFMPEG_avcodec_LIBRARY)
  set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} FFMPEG_avcodec_LIBRARY)
endif(FFMPEG_avcodec_LIBRARY)

find_library(FFMPEG_avutil_LIBRARY NAMES avutil-49 avutil aviplayavutil
  PATHS
  $ENV{FFMPEG_DIR}/lib
  /usr/lib
  /usr/local/lib
  /usr/lib64/lib
  /usr/local/lib64
)
if (FFMPEG_avutil_LIBRARY)
  set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} FFMPEG_avutil_LIBRARY)
endif(FFMPEG_avutil_LIBRARY)

find_library(FFMPEG_swscale_LIBRARY NAMES swscale-0 swscale
  PATHS
  $ENV{FFMPEG_DIR}/lib
  /usr/lib
  /usr/local/lib
  /usr/lib64/lib
  /usr/local/lib64
)
if (FFMPEG_swscale_LIBRARY)
  set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES} FFMPEG_swscale_LIBRARY)
endif(FFMPEG_swscale_LIBRARY)

find_path(FFMPEG_AV_INCLUDE_DIR avcodec.h
  $ENV{FFMPEG_DIR}/include/libavcodec
  /usr/include/libavcodec
  /usr/local/include/libavcodec
  /usr/lib64/include/libavcodec
  /usr/local/include/libavcodec
)

SET(FFMPEG_INCLUDE_DIR ${FFMPEG_AV_INCLUDE_DIR}/..)

if (FFMPEG_swscale_LIBRARY)
  set(FFMPEG_swscale_FOUND "YES" )
endif()

include_directories(${FFMPEG_INCLUDE_DIR})



set(VOLY_VIDEO_MODULE_HEADERS
  ffmpegrecorderworker.h
  framepool.h
  ffmpeg_encoder.hpp
  ffmpegrecorder.h
)

set(VOLY_VIDEO_MODULE_SOURCES
  framepool.cpp
  ffmpeg_encoder.cpp
  ffmpegrecorder.cpp
  ffmpegrecorderworker.cpp
)

add_definitions("-D__STDC_CONSTANT_MACROS")

#message("${FFMPEG_avcodec_LIBRARY} ${FFMPEG_avutil_LIBRARY} ${FFMPEG_swscale_LIBRARY

add_library(volyvideomodule SHARED ${VOLY_VIDEO_MODULE_HEADERS} ${VOLY_VIDEO_MODULE_SOURCES} ${VOLY_VIDEO_MODULE_MOC_SRC})
target_link_libraries(volyvideomodule
  Qt5::Core
  ${FFMPEG_avcodec_LIBRARY}
  ${FFMPEG_avutil_LIBRARY}
  ${FFMPEG_swscale_LIBRARY}

)

if(MSVC)
  install(TARGETS volyvideomodule
    RUNTIME DESTINATION "."
  COMPONENT Runtime
  )

  install(TARGETS volyvideomodule
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT libraries
  )
  install(FILES
    ${FFMPEG_avcodec_LIBRARY}
    ${FFMPEG_avutil_LIBRARY}
    ${FFMPEG_swscale_LIBRARY}
    DESTINATION "lib"
  COMPONENT deplibraries
  )
  file(GLOB VOLY_FFMPEG_RUNTIME_LIBRARIES $ENV{FFMPEG_DIR}/lib/*.dll)
  install(FILES ${VOLY_FFMPEG_RUNTIME_LIBRARIES}
    DESTINATION "."
    COMPONENT Runtime
  )
endif()
