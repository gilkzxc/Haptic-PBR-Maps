#pragma once

#include <random>

namespace radu{
namespace utils{

//https://channel9.msdn.com/Events/GoingNative/2013/rand-Considered-Harmful
// https://kristerw.blogspot.com/2017/05/seeding-stdmt19937-random-number-engine.html
class RandGenerator{
public:
    RandGenerator(unsigned int seed=0):
        // m_gen((std::random_device())()) //https://stackoverflow.com/a/29580889
        m_gen(seed) //start with a defined seed
        {

    }
    //returns a random float in the range [a,b], inclusive
    float rand_float(float a, float b) {
        std::uniform_real_distribution<float> distribution(a,b);
        return distribution(m_gen);
    }

    //returns a random float with a normal distribution with mean and stddev
    float rand_normal_float(float mean, float stddev) {
        std::normal_distribution<float> distribution(mean, stddev);
        return distribution(m_gen);
    }

    //returns a random int in the range between [a,b] inclusive
    int rand_int(int a, int b) {
        std::uniform_int_distribution<int> distribution(a,b);
        return distribution(m_gen);
    }

    //return a randomly bool with a probability of true of prob_true
    bool rand_bool(const float prob_true){
        std::bernoulli_distribution distribution(prob_true);
        return distribution(m_gen);
    }

    std::mt19937& generator(){
        return m_gen;
    }


private:
    std::mt19937 m_gen;
};

} //namespace utils
} //namespace radu
