/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_renderbin.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_RENDERBIN_HPP
#define GPUCAST_VOLUME_RENDERER_RENDERBIN_HPP

// header, system
#include <set>

// header, external

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/isosurface/fragment/renderchunk.hpp>

namespace gpucast {

class GPUCAST_VOLUME renderbin
{
public : // enums / typedefs

  typedef beziervolume::attribute_type          attribute_type;
  typedef attribute_type::value_type            value_type;
  typedef gpucast::math::interval<value_type>             interval_type;

  typedef std::vector<renderchunk_ptr>          renderchunk_container;
  typedef renderchunk_container::const_iterator renderchunk_const_iterator;

public : // ctors/dtor

  renderbin                               ( );
  renderbin                               ( interval_type const& range );
  renderbin                               ( renderbin const& );

  renderbin&                    operator= ( renderbin const& );

  void                          swap      ( renderbin& other );

  ~renderbin                              ();

public : // operators

public : // methods

  interval_type const&          range     () const;
  void                          range     ( interval_type const& range );

  std::size_t                   chunks    () const;      

  renderchunk_const_iterator    begin     () const;
  renderchunk_const_iterator    end       () const;

  int                           indices   () const;
  int                           baseindex () const;

  void                          indices   ( int i );
  void                          baseindex ( int i );
                          
  void                          insert    ( renderchunk_ptr const& );

private : // attributes

  interval_type                 _range;
  renderchunk_container         _chunks;
  int                           _baseindex;
  int                           _indices;
};

bool operator==( renderbin const& lhs, renderbin const& rhs );
bool operator!=( renderbin const& lhs, renderbin const& rhs );
bool operator> ( renderbin const& lhs, renderbin const& rhs );
bool operator< ( renderbin const& lhs, renderbin const& rhs );

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_RENDERBIN_HPP