/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_fraglist_raycasting.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_FRAGLIST_RAYCASTING_HPP
#define GPUCAST_VOLUME_RENDERER_FRAGLIST_RAYCASTING_HPP

// header, system

// header, external
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/texture1d.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/primitives/plane.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/framebufferobject.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/isosurface/fragment/fragmentlist_generator.hpp>

// type fwd 
struct cudaGraphicsResource;

namespace gpucast 
{

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME isosurface_renderer_fraglist_raycasting : public fragmentlist_generator
{
public : // enums, typedefs

  typedef fragmentlist_generator basetype;

public : // c'tor / d'tor

  isosurface_renderer_fraglist_raycasting          ( int argc, char** argv );
  virtual ~isosurface_renderer_fraglist_raycasting ();

public : // methods

  virtual void                            init                            ( drawable_ptr const&     object,
                                                                            std::string const&      attribute_name );

  virtual bool                            initialize                      ( std::string const& attribute_name ) = 0;

  virtual void                            draw                            ();
  /* virtual */ void                      recompile                       ();
  /* virtual */ void                      resize                          ( int width, int height );
                                   
  /* virtual */ void                      set_external_texture            ( std::shared_ptr<gpucast::gl::texture2d> const& texture );

  void                                    register_cuda_resources          ();
  void                                    unregister_cuda_resources       ();

  virtual void                            raycast_fragment_lists          () = 0;

private : // auxilliary methods                                           
                                                                          
  void                                    _init_glresources               ();
  void                                    _init_shader                    ();
  void                                    _init_platform                  ();

  void                                    _clean_textures                 ();

  void                                    _cuda_register_gl_image         ( struct cudaGraphicsResource **resource, GLuint image, GLenum target, unsigned int flags );
  void                                    _cuda_register_gl_buffer        ( struct cudaGraphicsResource **resource, GLuint buffer, unsigned int flags );

  void                                    _update_matrices                ();
  void                                    _draw_result                    ();
  
private : // attributes

  // GL ressources
  std::shared_ptr<gpucast::gl::program>            _clear_textures_pass;
  std::shared_ptr<gpucast::gl::program>            _quad_pass;
  std::shared_ptr<gpucast::gl::program>            _intersect_surface_pass;

  std::shared_ptr<gpucast::gl::framebufferobject>  _fbo;
  texture2d_ptr                               _depthattachment;
  texture2d_ptr                               _colorattachment;
  std::shared_ptr<gpucast::gl::sampler>            _no_interpolation;
  std::shared_ptr<gpucast::gl::sampler>            _linear_interpolation;

  bool                                        _initialized_gl;

protected :

  // CUDA resources 
  bool                                        _cuda_resources_mapped;
  cudaGraphicsResource*                       _cuda_colorbuffer;
  cudaGraphicsResource*                       _cuda_depthbuffer;
  cudaGraphicsResource*                       _cuda_headpointer;
  cudaGraphicsResource*                       _cuda_fragmentcount;
  cudaGraphicsResource*                       _cuda_indexlist;
  cudaGraphicsResource*                       _cuda_matrixbuffer;
  cudaGraphicsResource*                       _cuda_allocation_grid;

  cudaGraphicsResource*                       _cuda_surface_data_buffer;
  cudaGraphicsResource*                       _cuda_surface_points_buffer;
  cudaGraphicsResource*                       _cuda_volume_data_buffer;
  cudaGraphicsResource*                       _cuda_volume_points_buffer;
  cudaGraphicsResource*                       _cuda_attribute_data_buffer;
  cudaGraphicsResource*                       _cuda_attribute_points_buffer;

  cudaGraphicsResource*                       _cuda_external_texture;

  std::shared_ptr<gpucast::gl::arraybuffer>        _matrixbuffer;
};

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_FRAGLIST_RAYCASTING_HPP
