find_path(Clipper_INCLUDE_DIR polyclipping/clipper.hpp)
find_library(Clipper_LIBRARIES polyclipping)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Clipper Clipper_INCLUDE_DIR Clipper_LIBRARIES)
