/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : uid.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_UID_HPP
#define GPUCAST_CORE_UID_HPP

// header, system

// header, exernal

// header, project
#include <gpucast/core/gpucast.hpp>



namespace gpucast {

class GPUCAST_CORE uid
{
private :

  uid()
    : _uid_map(),
      _mtx()
  {}

  uid(uid const& other);

  uid& operator=(uid const&);

public : 

  static unsigned generate( std::string const& category )
  {
    static uid the_uid;
    boost::mutex::scoped_lock lck(the_uid._mtx);
    
    if ( the_uid._uid_map.count(category) )
    {
      return ++the_uid._uid_map.find(category)->second;
    } else {
      unsigned const start_uid = 1;
      the_uid._uid_map.insert(std::make_pair(category, start_uid));
      return start_uid;
    }
  }

  ~uid()
  {}

private :

  std::unordered_map<std::string, unsigned>	_uid_map;
  boost::mutex									              _mtx;
};

} // namespace gpucast

#endif // GPUCAST_CORE_UID_HPP