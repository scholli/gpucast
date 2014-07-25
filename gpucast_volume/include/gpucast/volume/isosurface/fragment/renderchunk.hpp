/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_renderchunk.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_RENDERCHUNK_HPP
#define GPUCAST_VOLUME_RENDERER_RENDERCHUNK_HPP

// header, system
#include <vector>

// header, external
#include <gpucast/math/interval.hpp>

#include <memory>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/beziervolume.hpp>


namespace gpucast {
  
struct renderchunk;
typedef std::shared_ptr<renderchunk> renderchunk_ptr;

///////////////////////////////////////////////////////////////////////////////
struct GPUCAST_VOLUME renderchunk 
{
  // typedef
  typedef beziervolume::attribute_type  attribute_type;
  typedef attribute_type::value_type    value_type;
  typedef gpucast::math::interval<value_type>     interval_type;

  renderchunk() 
  {}

  template <typename container_t>
  renderchunk ( container_t const& vertex_indices, gpucast::math::interval<value_type> const& i, bool outer_boundary )
    : indices ( vertex_indices.begin(), vertex_indices.end() ),
      range   ( i.minimum(), i.maximum(), gpucast::math::included, gpucast::math::included ),
      outer   ( outer_boundary )
  {}

  inline int size () 
  {
    return int ( indices.size() );
  }

  // methods
  inline bool operator< ( renderchunk const& rhs ) const
  { 
    return range.minimum() < rhs.range.minimum();
  }

  inline void write ( std::ostream& os ) const
  {
    std::size_t nindices = indices.size();
    os.write ( reinterpret_cast<char const*> (&nindices), sizeof(std::size_t) );
    os.write ( reinterpret_cast<char const*> (&indices.front()), sizeof(int) * nindices );

    value_type amin = range.minimum();
    value_type amax = range.maximum();

    os.write ( reinterpret_cast<char const*> (&amin), sizeof(value_type) );
    os.write ( reinterpret_cast<char const*> (&amax), sizeof(value_type) );
    os.write ( reinterpret_cast<char const*> (&outer), sizeof(bool) );
  }

  inline void read  ( std::istream& is )
  {
    std::size_t nindices;
    is.read ( reinterpret_cast<char*> (&nindices), sizeof(std::size_t) );
    indices.resize(nindices);

    is.read ( reinterpret_cast<char*> (&indices.front()), sizeof(int) * nindices );

    value_type amin;
    value_type amax;

    is.read ( reinterpret_cast<char*> (&amin), sizeof(value_type) );
    is.read ( reinterpret_cast<char*> (&amax), sizeof(value_type) );

    is.read ( reinterpret_cast<char*> (&outer), sizeof(bool) );
  }

  // attributes 
  std::vector<int>          indices;
  interval_type             range;
  bool                      outer;
};

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_RENDERCHUNK_HPP