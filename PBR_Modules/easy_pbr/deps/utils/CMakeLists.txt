# cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # we need at least 3.11 because we use the FetchContent feature to manage dependencies and at least 3.14 because we use FetchContent_MakeAvailable
cmake_minimum_required(VERSION 2.8...3.22) 

project(utils)

include(FetchContent)

###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -Wall -O3") #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD 17) #needed also by the pybind so it can build itself in c++14 mode so we can use py:overload_cast



######   PACKAGES   ############################################################
find_package(Boost REQUIRED COMPONENTS system thread filesystem)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
FetchContent_Declare(
  loguru
  GIT_REPOSITORY https://github.com/RaduAlexandru/loguru.git
)
##fetched content make availalbe. Order is important. It should go from the one with lest dependencies to the one with most. So if A depend on B, then MakeAvailable should have B first
FetchContent_MakeAvailable( loguru )


###   INCLUDES   #########################################################
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${loguru_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIR})


# ###   SOURCES   #################################################################
# set(MY_SRC
#     src/ColormapMngr.cxx
# )


# add_library( utils ${MY_SRC} )


# ###   EXECUTABLE   #######################################
# # add_executable(surfel_renderer ${CMAKE_SOURCE_DIR}/src/null.cxx  )
# # set_target_properties(surfel_renderer PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
# # add_executable(easy_pbr  ${EXTERN_SRC} ${IMGUI_SRC} src/main.cxx)


# ###   LIBS   ###############################################
# set(LIBS ${LIBS} Eigen3::Eigen  ${Boost_LIBRARIES} ${OpenCV_LIBS} )


# target_link_libraries(utils ${LIBS} )


###   TEST   ###############################################
add_subdirectory (test)
enable_testing ()
add_test (NAME profiler_timing COMMAND profiler_timing)
add_test (NAME eigen_filtering COMMAND eigen_filtering)
#running make test doesnt give any output so we need ctest https://stackoverflow.com/a/49942787 and from https://stackoverflow.com/a/48937782
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
    --force-new-ctest-process
    # --verbose
    --output-on-failure
)
