/*******************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : mesh3d_loader.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MESH3D_LOADER_HPP
#define GPUCAST_MESH3D_LOADER_HPP

// header, system
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME mesh3d_loader 
{
public : // c'tor / d'tor

  mesh3d_loader  ();
  ~mesh3d_loader ();

public : // methods

  bool load           ( std::string const& filename, 
                        std::shared_ptr<nurbsvolumeobject> const& ptr,
                        bool displace = false,
                        std::string = "" );
  bool load           ( std::string const& filename, std::shared_ptr<nurbssurfaceobject> const& ptr );

private : // methods

  mesh3d_loader& operator=(mesh3d_loader const& other) = delete;
  mesh3d_loader(mesh3d_loader const& other) = delete;

  bool _parse         ( std::string const& filename, std::shared_ptr<nurbsvolumeobject> const& ptr,
                        bool displace, std::string const& );
  bool _parse         ( std::string const& filename, std::shared_ptr<nurbssurfaceobject> const& ptr );

  bool _load_volumes  ( std::shared_ptr<nurbsvolumeobject> const& ptr,
                        bool displace, std::string const& );

  bool _load_surfaces ( std::shared_ptr<nurbssurfaceobject> const& ptr );

private : // attributes

  float                                            _geometry_scale;
  float                                            _attribute_scale;
                                                   
  std::string                                      _geometry_basename;
  std::string                                      _solution_basename;
                                                   
  std::unordered_map<std::string, unsigned>      _solution_attributes;       // name, dims
  std::map<std::string, std::string>               _geometry_solution_mapping;
  std::map<std::string, std::array<unsigned,4>>  _geometry_structure_mapping;
  std::array<std::set<unsigned>,4>               _geometry_structure;
  
  
};


} // namespace gpucast

#endif // GPUCAST_MESH3D_LOADER_HPP
