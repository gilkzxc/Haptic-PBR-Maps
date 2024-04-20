#define PROFILER_IMPLEMENTATION 1
#include "Profiler.h"


#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

void sleep_millisec(const int milliseconds){
    std::this_thread::sleep_for (std::chrono::milliseconds(milliseconds));
}

void start_stop(){
    //start-stop
    int duration=10;

    TIME_START("test1");
    sleep_millisec(duration);
    TIME_END("test1");

    float estimated=ELAPSED("test1");
    float error=std::fabs(estimated-duration);
    if( error>0.1 ){
        LOG(FATAL) << "The error in the estimate is too large. The duration should be " << duration << " but the estimated is " << estimated << " which is an error of " << error << " millisecond";
    }
}


void start_pause_stop(){
    //start-pause-stop

    int duration_part1=5;
    TIME_START("test2");
    sleep_millisec(duration_part1);
    TIME_PAUSE("test2");

    //do some untimed work
    sleep_millisec(40);

    int duration_part2=70;
    TIME_START("test2");
    sleep_millisec(duration_part2);
    TIME_END("test2");

    PROFILER_PRINT();

    //estimated should be the sum of the two regions that were timed
    float estimated=ELAPSED("test2");
    float error=std::fabs(estimated-duration_part1-duration_part2);
    if( error>0.1 ){
        LOG(FATAL) << "The error in the estimate is too large. The duration should be " << duration_part1+duration_part2 << " but the estimated is " << estimated << " which is an error of " << error << " millisecond";
    }
}

void scope(){

    int duration=25;
    {
        TIME_SCOPE("test3"); //time starts running here until it runs out of scope
        sleep_millisec(duration);
    } //scope finishes and the timer ends

    float estimated=ELAPSED("test3");
    float error=std::fabs(estimated-duration);
    if( error>0.1 ){
        LOG(FATAL) << "The error in the estimate is too large. The duration should be " << duration << " but the estimated is " << estimated << " which is an error of " << error << " millisecond";
    }

}

int main(int argc, char *argv[]) {

    start_stop();
    start_pause_stop();
    scope();


    return 0;

}
