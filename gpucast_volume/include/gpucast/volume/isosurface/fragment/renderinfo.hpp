/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_renderinfo.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_RENDERINFO_HPP
#define GPUCAST_VOLUME_RENDERER_RENDERINFO_HPP

// header, system
#include <set>

// header, external
#include <gpucast/math/interval.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/beziervolume.hpp>
#include <gpucast/volume/isosurface/fragment/renderbin.hpp>


namespace gpucast {

class split_heuristic;

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME renderinfo
{
public : // enums / typedefs

  typedef beziervolume::attribute_type        attribute_type;
  typedef attribute_type::value_type          value_type;
  typedef gpucast::math::interval<value_type>           interval_type;

  typedef std::vector<renderbin>              renderbin_container;
  typedef renderbin_container::const_iterator renderbin_const_iterator;

public : // ctors/dtor

  renderinfo                                  ();
  renderinfo                                  ( interval_type const & );
  ~renderinfo                                 ();

public : // operators

public : // methods

  std::size_t                       size          () const;
  void                              clear         ();

  std::vector<int>::const_iterator  begin         () const;
  std::vector<int>::const_iterator  end           () const;

  interval_type const&              range         ( ) const;
  void                              range         ( interval_type const& );
  
  void                              insert        ( renderchunk_ptr const& );

  void                              optimize      ( split_heuristic const& h );
  void                              serialize     ( );

  std::size_t                       renderbins_size ( ) const;
  renderbin_const_iterator          renderbin_begin ( ) const;
  renderbin_const_iterator          renderbin_end   ( ) const;

  bool                              get_renderbin ( value_type isovalue, int& index, int& size ) const;
  void                              get_outerbin  ( int& index, int& size ) const;

  void                              write         ( std::ostream& os ) const;
  void                              read          ( std::istream& is );

private : // attributes

  interval_type                     _range;

  renderbin                         _outerbin;    // outer surfaces are rendered anyway
  renderbin_container               _renderbins;  // other chunks are rendered dependent on 
  
  std::vector<int>                  _indices;
};

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_RENDERINFO_HPP