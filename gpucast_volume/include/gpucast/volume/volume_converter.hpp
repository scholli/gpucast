/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_converter.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_CONVERTER_HPP
#define GPUCAST_VOLUME_CONVERTER_HPP

// header, system

// header, external
#include <memory>
#include <boost/multi_array.hpp>

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziervolume.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/beziervolume.hpp>


template class GPUCAST_VOLUME gpucast::math::beziervolume<gpucast::math::point<double,3> >;

namespace gpucast {

// forward declarations
class beziervolumeobject;
class nurbsvolumeobject;
class nurbsvolume;

class GPUCAST_VOLUME volume_converter
{
public : // enums / typedefs

  typedef boost::multi_array<unsigned, 3>   unsigned_array3d_t;
  typedef boost::multi_array<unsigned, 4>   unsigned_array4d_t;

public : // ctors/dtor

  volume_converter();
  ~volume_converter();

public : // operators

  void convert    ( std::shared_ptr<nurbsvolumeobject> const& ns, std::shared_ptr<beziervolumeobject> const& bs );

private : // methods

  volume_converter(volume_converter const&) = delete;
  volume_converter& operator=(volume_converter const&) = delete;

  void                _convert              ( nurbsvolume const& nurbsvolume );
  void                _fetch_task           ();
  unsigned            _compute_surface_id   ( unsigned_array4d_t const& ids,    // unique surface ids
                                              unsigned u,                       // # bezier element in u
                                              unsigned v,                       // # bezier element in v
                                              unsigned w,                       // # bezier element in w
                                              unsigned nelements_u,             // total number of bezier elements in u
                                              unsigned nelements_v,             // total number of bezier elements in v
                                              unsigned nelements_w,             // total number of bezier elements in w
                                              beziervolume::boundary_t surface  // surface type
                                            ) const;

  unsigned            _compute_neighbor_id  ( unsigned_array3d_t const& ids,    // unique volume ids
                                              unsigned u,                       // # bezier element in u
                                              unsigned v,                       // # bezier element in v
                                              unsigned w,                       // # bezier element in w
                                              unsigned nelements_u,             // total number of bezier elements in u
                                              unsigned nelements_v,             // total number of bezier elements in v
                                              unsigned nelements_w,             // total number of bezier elements in w
                                              beziervolume::boundary_t surface  // surface type
                                            ) const;

  void                _identify_neighbors   ();

  void                _generate_volume_ids  ( unsigned nelements_u, unsigned nelements_v, unsigned nelements_w, unsigned_array3d_t& ) const;
  void                _generate_surface_ids ( unsigned nelements_u, unsigned nelements_v, unsigned nelements_w, unsigned_array4d_t& ) const;
  void                _extract_adjacency    ( unsigned_array3d_t const& ids, 
                                              unsigned u, 
                                              unsigned v, 
                                              unsigned w, 
                                              unsigned nelements_u, 
                                              unsigned nelements_v, 
                                              unsigned nelements_w,
                                              beziervolume::adjacency_map& m ) const;
    

private : // attributes

  class impl_t;
  impl_t* _impl;

};

} // namespace gpucast

#endif // GPUCAST_VOLUME_CONVERTER_HPP
