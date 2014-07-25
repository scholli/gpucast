/*******************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : xml_loader.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_XML_LOADER_HPP
#define GPUCAST_XML_LOADER_HPP

// header, system
#include <string>
#include <vector>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>

namespace gpucast {


///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME xml_loader 
{
public : // c'tor / d'tor

  xml_loader  ();
  ~xml_loader ();

public : // methods

  bool load           ( std::string const& filename, 
                        std::shared_ptr<nurbsvolumeobject> const& ptr,
                        bool displace = false,
                        std::string = "" );

private : // non-copyable

  xml_loader(xml_loader const& other) = delete;
  xml_loader& operator=(xml_loader const& other) = delete;

private: // methods

  bool _parse         ( std::string const& filecontent, 
                        std::shared_ptr<nurbsvolumeobject> const& ptr,
                        bool displace,
                        std::string const& attrib );

private : // attributes

};


} // namespace gpucast

#endif // GPUCAST_XML_LOADER_HPP
