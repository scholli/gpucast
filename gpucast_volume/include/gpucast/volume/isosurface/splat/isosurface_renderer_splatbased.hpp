/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_clsplat.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_ISOSURFACE_RENDERER_HSPLAT_HPP
#define GPUCAST_ISOSURFACE_RENDERER_HSPLAT_HPP

// header, system

// header, external
#include <memory>
#include <boost/unordered_map.hpp>

#include <gpucast/gl/program.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/volume_renderer.hpp>



namespace gpucast 
{

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME isosurface_renderer_splatbased : public volume_renderer
{
public : // enums, typedefs

public : // c'tor / d'tor

  isosurface_renderer_splatbased    ( int argc, char** argv );
  ~isosurface_renderer_splatbased   ();

public : // methods

  /* virtual */ void                  clear                 ();

  virtual void                        init                  ( drawable_ptr const& object,
                                                              std::string const&  attribute_name );

  /* virtual */ void                  draw                  ();

  /* virtual */ void                  transform             ( gpucast::gl::matrix4f const& m );

  /* virtual */ void                  compute_nearfar       ();

  /* virtual */ void                  recompile             ();

  void                                unregister_cuda_resources ();

  /* virtual */ void                  write ( std::ostream& os ) const;
  /* virtual */ void                  read  ( std::istream& is );

private : // auxilliary methods

  void                                _init_shader          ();

private : // attributes 

};

} // namespace gpucast

#endif // GPUCAST_ISOSURFACE_RENDERER_HSPLAT_HPP
