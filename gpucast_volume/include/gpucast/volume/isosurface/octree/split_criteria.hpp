/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : split_criteria.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_SPLIT_CRITERIA_HPP
#define GPUCAST_SPLIT_CRITERIA_HPP

// header, system

// header, project

namespace gpucast {

// forward declaration
class ocnode;

class split_criteria 
{
public : 

  split_criteria();
  virtual ~split_criteria();

public : 

  virtual bool operator() ( ocnode const& ) const = 0;

};

} // namespace gpucast

#endif // GPUCAST_OCSPLIT_CRITERIA_HPP
