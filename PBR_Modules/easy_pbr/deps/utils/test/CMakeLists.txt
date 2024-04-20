cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # we need at least 3.11 because we use the FetchContent feature to manage dependencies and at least 3.14 because we use FetchContent_MakeAvailable

project(TEST)


###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -Wall -O3") #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD 17) #needed also by the pybind so it can build itself in c++14 mode so we can use py:overload_cast



###   EXECUTABLE   #######################################
set(LIBS -lpthread -ldl) #because loguru needs them
add_executable(profiler_timing ${PROJECT_SOURCE_DIR}/profiler/timing.cxx  )
target_link_libraries(profiler_timing ${LIBS} )
add_executable(eigen_filtering ${PROJECT_SOURCE_DIR}/eigen/filtering.cxx  )
target_link_libraries(eigen_filtering ${LIBS} )
