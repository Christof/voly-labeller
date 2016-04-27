# - Try to find Clipper
# Once done this will define
#  Clipper_FOUND - System has Clipper
#  Clipper_INCLUDE_DIRS - The Clipper include directories
#  Clipper_LIBRARIES - The libraries needed to use Clipper
#  Clipper_DEFINITIONS - Compiler switches required for using Clipper

if (UNIX)
  find_package(PkgConfig)
  pkg_check_modules(PC_Clipper QUIET polyclipping.pc)
  set(Clipper_DEFINITIONS ${PC_Clipper_CFLAGS_OTHER})
endif()
  
  MESSAGE(STATUS "clipper dir: $ENV{CLIPPER_DIR}" )

  find_path(Clipper_INCLUDE_DIR polyclipping/clipper.hpp
    HINTS 
	${PC_Clipper_INCLUDEDIR} ${PC_Clipper_INCLUDE_DIRS}
    $ENV{CLIPPER_DIR}
	$ENV{CLIPPER_ROOT}
    PATH_SUFFIXES include libpolyclipping
  )

  find_library(Clipper_LIBRARY NAMES polyclipping libpolyclipping
    HINTS 
	${PC_Clipper_LIBDIR} ${PC_Clipper_LIBRARY_DIRS}
    $ENV{CLIPPER_DIR}
	$ENV{CLIPPER_ROOT}	
	PATH_SUFFIXES lib 
  )

  include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set Clipper_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Clipper  DEFAULT_MSG
  Clipper_LIBRARY Clipper_INCLUDE_DIR)

mark_as_advanced(Clipper_INCLUDE_DIR Clipper_LIBRARY)

set(Clipper_LIBRARIES ${Clipper_LIBRARY})
set(Clipper_INCLUDE_DIRS ${Clipper_INCLUDE_DIR})
