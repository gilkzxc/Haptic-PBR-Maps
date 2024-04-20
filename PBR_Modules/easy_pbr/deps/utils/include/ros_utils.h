#pragma once

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

// #include "StringTools.h"
#include <iostream>


namespace radu{
namespace utils{

template <typename T>
T getParamElseError ( ros::NodeHandle & nh, const std::string & paramName )
{
   T out;

   if ( ! nh.getParam( paramName, out ) )
   {
      ROS_ERROR_STREAM( "Parameter " << paramName << " could not be retrieved." );
   }

   ROS_INFO_STREAM ( paramName << ": " << std::to_string ( out ) );

   return out;
}

template <typename T>
T getParamElseThrow( ros::NodeHandle & nh, const std::string & paramName )
{
  T out;

  if ( ! nh.getParam( paramName, out ) )
  {
    std::cout << std::string("Parameter " + paramName + " could not be retrieved." ) << '\n';
    throw std::string("Parameter " + paramName + " could not be retrieved." );
  }

  ROS_INFO_STREAM( paramName << ": " << std::to_string( out ) );
  return out;
}

template <typename T>
T getParamElseDefault( ros::NodeHandle & nh, const std::string & paramName, const T & def )
{
  T out;

  if ( ! nh.getParam( paramName, out ) )
  {
    ROS_INFO_STREAM( paramName << ": Default value used (" << to_string( def ) << ")" );
    return def;
  };

  ROS_INFO_STREAM( paramName << ": " << std::to_string( out ) );

  return out;
}

template <typename T>
void getSingleMsgFromTopic( T * msg, const std::string & topicName, ros::NodeHandle & nh = ros::NodeHandle(),  const int freq = 20 )
{
  bool inited = false;

  typedef boost::shared_ptr< T const > TConstPtr;

  ros::Subscriber sub = nh.subscribe<T>(
    topicName, 1, [&] (const TConstPtr & m) { *msg = *m; inited = true; } );

  ros::Rate r(freq);

  while ( ros::ok() && ! inited )
  {
    ros::spinOnce();
    r.sleep();
  }
}


} //namespace utils
} //namespace radu
