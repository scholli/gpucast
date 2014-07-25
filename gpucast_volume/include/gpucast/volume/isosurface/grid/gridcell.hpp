/********************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : gridcell.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GRIDCELL_HPP
#define GPUCAST_GRIDCELL_HPP

// header, system

// header, external

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/isosurface/face.hpp>

#include <gpucast/math/interval.hpp>

namespace gpucast {

class GPUCAST_VOLUME gridcell
{
public : // typedefs / enums

  typedef gpucast::math::interval<beziervolume::attribute_type::value_type> attribute_interval_t;

public : // c'tor / d'tor

  gridcell();

public : // methods

  void clear  ();
  void add    ( face_ptr const& face );

  float attribute_min () const;
  float attribute_max () const;

  std::vector<face_ptr>::const_iterator begin() const;
  std::vector<face_ptr>::const_iterator end() const;

  std::size_t faces () const;

private : // auxilliary methods

private : // attributes

  attribute_interval_t    _range;
  std::vector<face_ptr>   _faces;

};

} // namespace gpucast

#endif // GPUCAST_GRIDCELL_HPP