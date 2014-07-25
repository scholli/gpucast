/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_interval_sampling.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_INTERVAL_SAMPLING_HPP
#define GPUCAST_VOLUME_RENDERER_INTERVAL_SAMPLING_HPP

// header, system

// header, external

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/isosurface/fragment/isosurface_renderer_fraglist_raycasting.hpp>

// type fwd 
struct cudaGraphicsResource;

namespace gpucast 
{

class split_heuristic;

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME isosurface_renderer_interval_sampling : public isosurface_renderer_fraglist_raycasting
{
public : // enums, typedefs

  typedef isosurface_renderer_fraglist_raycasting basetype;

public : // c'tor / d'tor

  isosurface_renderer_interval_sampling    ( int argc, char** argv );
  ~isosurface_renderer_interval_sampling   ();

public : // methods

  virtual void                            raycast_fragment_lists ();

  virtual bool                            initialize             ( std::string const& attribute_name );

private : // auxilliary methods                                           
                                                                 
  void                                    _create_proxy_geometry ( beziervolume const& v, std::vector<deferred_surface_header_write>& deferred_jobs );
  void                                    _build_convex_hull     ( beziervolume const& v, enum beziervolume::boundary_t face_type, deferred_surface_header_write& job );
  void                                    _build_paralleliped    ( beziervolume const& v, enum beziervolume::boundary_t face_type, deferred_surface_header_write& job );
                                                                 
private : // attributes

  proxy_type                              _proxy_type;
  boost::scoped_ptr<split_heuristic>      _bin_split_heuristic;

};

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_INTERVAL_SAMPLING_HPP
