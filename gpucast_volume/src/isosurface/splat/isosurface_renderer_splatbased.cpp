/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_clsplat.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/splat/isosurface_renderer_splatbased.hpp"

// header, system
#include <GL/glew.h>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/get_nearfar.hpp>
#include <gpucast/gl/util/timer.hpp>

// header, project
#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/uid.hpp>

// tmp
#include <boost/iterator/transform_iterator.hpp>
#include <gpucast/math/util/pair_adaptor.hpp>


#if WIN32
  #pragma warning(disable: 4512) // cl.hpp assignment operator could not be generated
  #pragma warning(disable: 4610) // can never be instantiated
#else
  #include <GL/glx.h>
#endif


namespace gpucast {

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_splatbased::isosurface_renderer_splatbased( int argc, char** argv )
    : volume_renderer (argc, argv)
  {}


  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_splatbased::~isosurface_renderer_splatbased()
  {}
  
  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_splatbased::init ( drawable_ptr const& object,
                                         std::string const&  attribute_name )
  {}


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_splatbased::clear ()
  {
    throw std::runtime_error("not implemented yet");
  }



  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  isosurface_renderer_splatbased::draw ()
  {
    throw std::runtime_error("not implemented yet");
  }

  /////////////////////////////////////////////////////////////////////////////
  void                      
  isosurface_renderer_splatbased::transform ( gpucast::math::matrix4f const& m )
  {
    throw std::runtime_error("not implemented yet");
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_splatbased::recompile ()
  {
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_splatbased::unregister_cuda_resources ()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_splatbased::_init_shader ()
  {
    volume_renderer::_init_shader();
  }

  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                  
  isosurface_renderer_splatbased::write ( std::ostream& os ) const
  {
    throw std::runtime_error("not implemented yet");
  }

  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                  
  isosurface_renderer_splatbased::read ( std::istream& is )
  {
    throw std::runtime_error("not implemented yet");
  }

} // namespace gpucast
