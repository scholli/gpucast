/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziersubvolume.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_BEZIERSUBVOLUME_HPP
#define GPUCAST_BEZIERSUBVOLUME_HPP

#include <gpucast/math/parametric/beziervolume.hpp>
#include <gpucast/math/parametric/point.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/beziervolume.hpp>



namespace gpucast {

struct beziersubvolume
{
  // typedefs
  typedef double                            value_type;
  typedef gpucast::math::point<value_type,3>          point_type; // for parameter space!
  //typedef std::shared_ptr<beziervolume>   volume_ptr;
  typedef beziervolume const*               volume_ptr;
  typedef beziervolume                      volume_type;

  // attributes
  volume_type                               volume;
  point_type                                uvw_min;
  point_type                                uvw_max;
  volume_ptr                                parent;

  beziersubvolume ( volume_type const& p )
    : volume  ( p ),
      uvw_min ( 0.0, 0.0, 0.0 ),
      uvw_max ( 1.0, 1.0, 1.0 ),
      parent  ( &p )
  {}

  template <typename iterator_t>
  void ocsplit ( iterator_t subvolume_insert_iterator ) const
  {
    beziervolume::array_type ocsplit = volume.split();

    // traverse splits and push into temporary subvolume container
    for ( std::size_t w = 0; w != 2; ++w )
    {
      for ( std::size_t v = 0; v != 2; ++v )
      {
        for ( std::size_t u = 0; u != 2; ++u )
        {
          point_type uvwmin (uvw_min[0] + u *     ( uvw_max[0] - uvw_min[0])/2,
                                                    uvw_min[1] + v * (uvw_max[1] - uvw_min[1])/2,
                                                    uvw_min[2] + w * (uvw_max[2] - uvw_min[2])/2 );
          point_type uvwmax (uvw_max[0] - (!u) * (  uvw_max[0] - uvw_min[0])/2,
                                                    uvw_max[1] - (!v) * (uvw_max[1] - uvw_min[1])/2,
                                                    uvw_max[2] - (!w) * (uvw_max[2] - uvw_min[2])/2 );
          beziersubvolume s;

          s.volume  = volume_ptr(new beziervolume(ocsplit[u][v][w]));
          s.uvw_min = uvwmin;
          s.uvw_max = uvwmax;
          s.parent  = parent;

          subvolume_insert_iterator = s;
        }
      }
    }
  }
};

} // namespace gpucast

#endif // GPUCAST_BEZIERSUBVOLUME_HPP
