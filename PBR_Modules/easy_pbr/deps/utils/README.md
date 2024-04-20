# utils

This is a loose collection of C++ 14 utilities that I tend to use between various projects. 

### Usage 
All libraries are self contained header file so one only needs to include the relevant `.h` file, e.g.:

	#include <utils/eigen_utils.h>

You also need to compile Loguru. Luckily, `loguru.hpp` is included in this repo, and to compile it you just need to add the following to one of your .cpp files:

	#define LOGURU_IMPLEMENTATION 1
	#include <loguru.hpp>

Make sure you compile with -std=c++11 -lpthread -ldl 

If you have CMake >=3.14 you can obtain all the utils by adding to your CMakeLists.txt:

    #Clones the utils
    FetchContent_Declare(
    utils
    GIT_REPOSITORY https://github.com/RaduAlexandru/utils.git )
    FetchContent_MakeAvailable(utils)
    
    include_directories(${utils_SOURCE_DIR}/include)

If you have an older version of CMake (<3.14) you can also add the utilities as a git submodule:

    git submodule add https://github.com/RaduAlexandru/utils.git extern/utils
and adding to your CMakeLists.txt: 

    add_subdirectory (utils)
    include_directories(extern/utils/include)
`

#### Documentation
This file (README.md) contains an overview of each library. Read the header for each library to learn more.

#### Tests
There is a very limited set of tests in the `tests/` folder.

# Stand-alone libraries

#### ColorMngr.h
Convenience functions for using colormaps and also generating random colors based on tinycolormap. Supports colormaps: 

| Name     | Sample                         |
|:--------:|:------------------------------:|
| Magma    | ![](https://github.com/yuki-koyama/tinycolormap/raw/master/docs/samples/Magma.png)    |
| Plasma   | ![](https://github.com/yuki-koyama/tinycolormap/raw/master/docs/samples/Plasma.png)   |
| Viridis  | ![](https://github.com/yuki-koyama/tinycolormap/raw/master/docs/samples/Viridis.png)  |

    ColorMngr color_mngr;
    Eigen::Vector3f color=color_mngr.plasma_color(0.1); 
    Eigen::MatrixXf colormap = color_mngr.viridis_colormap(); //matrix of 256x3

#### Profiler.h 
Allows for easy profiling of code with high precision timers: 

    #define PROFILER_IMPLEMENTATION 1
    #include "Profiler.h"
    
    TIME_START("timer")
    func()
    TIME_END("timer")
    
    {
        TIME_SCOPE("another_timer") //time starts running here until it runs out of scope
        work_inside_scope()
    } //scope finishes and the timer ends
    
    float millisecond=ELAPSED("mytime)
    PROFILER_PRINT(); // print mean, std_dev, min and max of all the timings recorded until now

#### RandGenerator.h
High quality random number generator which abstracts away some of the verbosity of the standard library. Example: 

    RandGenerator gen;
    float rand_uniform = gen.rand_float(0.0, 10.0); //random uniform number between 0 and 10
    float rand_normal = gen.rand_normal_float(0.0, 0.1); // normal with mean and std_dev
    bool coin = gen.rand_bool(0.5); //0.5 probability of true
    
#### eigen_utils.h
Useful functions to have for manipulating Eigen matrices (removing rows, columns, reindexing, etc). They are particularly useful for processing meshes which are represented as matrices of vertices V and faces F. Easy_PBR heavily uses this library for aiding geometry processing. 

#### string_utils.h 
Usefult functions manipulating string which includes but is not limited to:
- trimming
- splitting by delimiter 
- formatting

#### opencv_utils.h
Convenience functions for OpenCV images
    
#### numerical_utils.h
Various numerical functions for:
- interpolation and shaping functions functions (lerp, step, smoothstep)
- indexing utilities (3D indices XYZ -> linear index )
- number manipulation ( degree2radians, clamping etc)
   
#### Ringbuffer.h
General purpose, single threaded ringbuffer:
    
    Ringbuffer ring=ringbuffer<float,50> // contains a maximum of 50 elements
    ring.push(30);
    ring.push(10); 
    std::cout << "Last element added is " << ring.back(); //prints 10
 