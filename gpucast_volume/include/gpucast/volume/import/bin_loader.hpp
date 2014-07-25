/*******************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : bin_loader.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_BIN_LOADER_HPP
#define GPUCAST_BIN_LOADER_HPP

// header, system
#include <string>
#include <map>
#include <memory>

// header, project
#include <gpucast/volume/gpucast.hpp>

namespace gpucast {

class beziervolumeobject;

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME bin_loader 
{
public : // c'tor / d'tor

  bin_loader  ();
  ~bin_loader ();

public : // methods

  bool load           ( std::string const& filename, std::shared_ptr<beziervolumeobject> const& ptr );

private : // methods

  bin_loader(bin_loader const& other) = delete;
  bin_loader& operator=(bin_loader const&) = delete;

private : // attributes

};


} // namespace gpucast

#endif // GPUCAST_BIN_LOADER_HPP
