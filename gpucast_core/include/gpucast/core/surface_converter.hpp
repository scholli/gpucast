/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : surface_converter.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_SURFACE_CONVERTER_HPP
#define GPUCAST_CORE_SURFACE_CONVERTER_HPP

// header, system
#include <memory>

// header, external
#include <boost/noncopyable.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>


namespace gpucast {

// forward declarations
class beziersurfaceobject;
class nurbssurfaceobject;
class nurbssurface;

class GPUCAST_CORE surface_converter : boost::noncopyable
{
public : // ctors/dtor

  surface_converter();
  ~surface_converter();

public : // operators

  void convert    ( std::shared_ptr<nurbssurfaceobject> const& ns, std::shared_ptr<beziersurfaceobject> const& bs );

private : // methods

  void _convert    ( nurbssurface const& nurbssurface );
  void _fetch_task ();

private : // attributes

  class impl_t;
  impl_t* _impl;

};

} // namespace gpucast

#endif // GPUCAST_CORE_SURFACE_CONVERTER_HPP
