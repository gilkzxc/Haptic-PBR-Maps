#pragma once

#include <vector>

#include <Eigen/Geometry>

// //loguru
#include <loguru.hpp>

namespace radu{
namespace utils{

// Converts degrees to radians.
inline float degrees2radians(float angle_degrees){
    return  (angle_degrees * M_PI / 180.0);
}

// Converts radians to degrees.
inline float radians2degrees(float angle_radians){
    return (angle_radians * 180.0 / M_PI);
}
//clamp a value between a min and a max
template <class T>
inline T clamp(const T val, const T min, const T max){
    return std::min(std::max(val, min),max);
}

//Best answer of https://stackoverflow.com/questions/5731863/mapping-a-numeric-range-onto-another
inline float map(const float input, const float input_start,const float input_end, const float output_start, const float output_end) {
    //we clamp the input between the start and the end
    float input_clamped=clamp(input, input_start, input_end);
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start);
}

//smoothstep like in glsl https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/smoothstep.xhtml
inline float smoothstep(const float edge0, const float edge1, const float x){
    CHECK(edge0 < edge1) << "The GLSL code for smoothstep only allows a transition from a lower number to a bigger one. Didn't have bother to modify this.";
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0 - 2.0 * t);
}

//FROM Fast and Funky 1D Nonlinear Transformations:  https://www.youtube.com/watch?v=mr5xkf6zSzk
//Cubic (3d degree) Bezier through A,B,C,D where A(start) and D(end) are assumed to be 1
inline float normalized_bezier(float B, float C, float t){
    CHECK(t<=1.0 && t>=0.0) << "t must be in range [0,1]";

    float s = 1.0f - t;
    float t2 = t*t;
    float s2 = s*s;
    float t3 = t2*2;
    return (3.0*B*s2*t) + (3.0*C*s*t2) + t3;
}


//Needed because % is not actually modulo in c++ and it may yield unexpected valued for negative numbers
//https://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c
inline int mod(int k, int n) {
    return ((k %= n) < 0) ? k+n : k;
}

//https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

inline bool XOR(bool a, bool b)
{
    return (a + b) % 2;
}

inline int next_power_of_two(int x) { // https://github.com/LWJGL/lwjgl3-wiki/wiki/2.6.1.-Ray-tracing-with-OpenGL-Compute-Shaders-(Part-I)
  x--;
  x |= x >> 1; // handle 2 bit numbers
  x |= x >> 2; // handle 4 bit numbers
  x |= x >> 4; // handle 8 bit numbers
  x |= x >> 8; // handle 16 bit numbers
  x |= x >> 16; // handle 32 bit numbers
  x++;
  return x;
}

//if the value gets bigger than the max it wraps back from 0, if its smaller than 0 it also wrap back from max
template <class T>
inline T wrap(const T val, const T max){
    T new_val = val;

    while(new_val >= max) new_val = (new_val - max);
    while(new_val < 0) new_val = (new_val + max);

    return new_val;
}

//from a 2D coordinate, get a linear one, with wraping
template <class T>
inline T idx_2D_to_1D_wrap(const T x, const T y, const T width, const T height){

    T x_wrap=wrap(x,width);
    T y_wrap=wrap(y,height);

    return y_wrap*width +x_wrap;
}

//from a 2D coordinate, get a linear one, without wrapping (thows error if accessing out of bounds)
template <class T>
inline T idx_2D_to_1D(const T x, const T y, const T width, const T height){
    CHECK(x<width) << "x is accessing out of bounds of the width. x is " << x << " width is " << width;
    CHECK(y<height) << "y is accessing out of bounds of the height. y is " << y << " height is " << height;

    T idx= y*width + x;
    CHECK(idx<width*height) << "idx will access out of bounds. idx is " << idx << " width*height is " << width*height;

    return idx;
}


//from a 1D coordinate, get a 2D one
template <class T>
inline Eigen::Vector2i idx_1D_to_2D(const T idx, const T width, const T height){

    T x=idx%width;
    T y=idx/width;

    CHECK(x<width) << "x is accessing out of bounds of the width. x is " << x << " width is " << width;
    CHECK(y<height) << "y is accessing out of bounds of the height. y is " << y << " height is " << height;

    Eigen::Vector2i pos;
    pos.x()=x;
    pos.y()=y;

    return pos;
}

//from a 1D coordinate, get a 3D one assuming that things are stored in memory in order zyx where x is the fastest changing dimension and z is the slowest
//output is a idx in xyz with respect to the origin of the grid
inline Eigen::Vector3i idx_1D_to_3D(const int idx, const Eigen::Vector3i& grid_sizes){

    // int z = idx / (grid_sizes.x() *grid_sizes.y() ) ; //we do a full z channel when we sweapped both x and y
    // int y = z % grid_sizes.y();
    // int x = y % grid_sizes.x();

    //http://www.alecjacobson.com/weblog/?p=1425
    int x = idx % grid_sizes.x();
    int y = (idx - x)/grid_sizes.x() % grid_sizes.y();
    int z = ((idx - x)/grid_sizes.x()-y)/ grid_sizes.y();

    CHECK(x<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << x << " x span is " << grid_sizes.x() ;
    CHECK(y<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << y << " y span is " << grid_sizes.y();
    CHECK(z<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << z << " y span is " << grid_sizes.z();

    Eigen::Vector3i pos;
    pos.x()=x;
    pos.y()=y;
    pos.z()=z;

    return pos;
}

//from a 3D coordinate, get a 1D one assuming that things are stored in memory in order zyx where x is the fastest changing dimension and z is the slowest
//input is a index in format xyz with respect to the origin of the grid
inline int idx_3D_to_1D(const Eigen::Vector3i pos, const Eigen::Vector3i& grid_sizes){

    //http://www.alecjacobson.com/weblog/?p=1425
    int index = pos.x() + grid_sizes.x()*(pos.y()+grid_sizes.y()*pos.z());

    CHECK(pos.x()<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << pos.x() << " x span is " << grid_sizes.x() ;
    CHECK(pos.y()<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << pos.y() << " y span is " << grid_sizes.y();
    CHECK(pos.z()<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << pos.z() << " y span is " << grid_sizes.z();

    return index;
}

//from a 1D coordinate, get a 4D one assuming that things are stored in memory in order wzyx where x is the fastest changing dimension and w is the slowest
//output is a idx in xyzw with respect to the origin of the grid
inline Eigen::Vector4i idx_1D_to_4D(const int idx, const Eigen::Vector4i& grid_sizes){

    // int z = idx / (grid_sizes.x() *grid_sizes.y() ) ; //we do a full z channel when we sweapped both x and y
    // int y = z % grid_sizes.y();
    // int x = y % grid_sizes.x();

    //http://www.alecjacobson.com/weblog/?p=1425
    int x = idx % grid_sizes.x();
    int y = (idx - x)/grid_sizes.x() % grid_sizes.y();
    int z = ((idx - x)/grid_sizes.x()-y) % grid_sizes.z();
    int w = (((idx - x)/grid_sizes.x()-y)-z)/ grid_sizes.z();

    CHECK(x<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << x << " x span is " << grid_sizes.x() ;
    CHECK(y<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << y << " y span is " << grid_sizes.y();
    CHECK(z<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << z << " y span is " << grid_sizes.z();
    CHECK(w<grid_sizes.w()) << "w is accessing out of bounds of the w grid size. w is " << z << " w span is " << grid_sizes.w();

    Eigen::Vector4i pos;
    pos.x()=x;
    pos.y()=y;
    pos.z()=z;
    pos.w()=w;

    return pos;
}

//from a 4D coordinate, get a 1D one assuming that things are stored in memory in order wzyx where x is the fastest changing dimension and w is the slowest
inline int idx_4D_to_1D(const Eigen::Vector4i pos, const Eigen::Vector4i& grid_sizes){

    //http://www.alecjacobson.com/weblog/?p=1425
    int index = pos.x() + grid_sizes.x()*(pos.y()+grid_sizes.y()*(pos.z() + grid_sizes.z()*pos.w())  );

    CHECK(pos.x()<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << pos.x() << " x span is " << grid_sizes.x() ;
    CHECK(pos.y()<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << pos.y() << " y span is " << grid_sizes.y();
    CHECK(pos.z()<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << pos.z() << " y span is " << grid_sizes.z();
    CHECK(pos.w()<grid_sizes.w()) << "w is accessing out of bounds of the w grid size. w is " << pos.w() << " w span is " << grid_sizes.w();

    return index;
}

// To return char for a value. For example '2'
// is returned for 2. 'A' is returned for 10. 'B'
// for 11
// Based on https://www.geeksforgeeks.org/convert-base-decimal-vice-versa/
inline char reVal(int num) {
    if (num >= 0 && num <= 9)
        return (char)(num + '0');
    else
        return (char)(num - 10 + 'A');
}

//go from decimat base to any other base and returns the digit of that base as a std vector. Based on https://www.geeksforgeeks.org/convert-base-decimal-vice-versa/
//useful for creating uniform boxel grids in any dimension where vertices are defined as (0,0), (0,1), (1,0), (1,1) etc. Look at misc_utils/PermutoLatticePlotter for details
inline std::vector<int> convert_decimal_to_base(const int num, const int base) {
    // int index = 0;  // Initialize index of result

    int val=num;
    std::vector<int> digits;
    // Convert input number is given base by repeatedly
    // dividing it by base and taking remainder
    while (val > 0) {
        digits.push_back( val%base );
        val /= base;
    }

    // Reverse the result
    std::reverse(digits.begin(), digits.end());

    return digits;
}

} //namespace utils
} //namespace radu
