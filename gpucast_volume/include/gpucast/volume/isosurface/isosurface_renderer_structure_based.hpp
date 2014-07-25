/********************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_structure_based.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_ISOURFACE_RENDERER_STRUCTURE_BASED_HPP
#define GPUCAST_ISOURFACE_RENDERER_STRUCTURE_BASED_HPP

// header, system
#include <list>

// header, external
#include <memory>
#include <unordered_map>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/volume_renderer.hpp>

#include <gpucast/volume/isosurface/renderconfig.hpp>
#include <gpucast/volume/isosurface/face.hpp>

#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/primitives/plane.hpp>

#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/program.hpp>

#include <gpucast/gl/math/vec4.hpp>

// type fwd 
struct cudaGraphicsResource;

namespace glpp {
  class program;
  class texturebuffer;
  class sampler;
  class texture2d;
  class framebufferobject;
  class plane;
  class cube;
  class renderbuffer;
};

namespace gpucast 
{

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class GPUCAST_VOLUME isosurface_renderer_structure_based : public volume_renderer
{
public : // enums, typedefs

public : // c'tor / d'tor

  isosurface_renderer_structure_based            ( int argc, char** argv );
  virtual ~isosurface_renderer_structure_based   ();

public : // methods

  void                            init                      ( drawable_ptr const& object,
                                                              std::string const& attribute_name );

  /* virtual */ void              clear                     ();

  void                            raygeneration             ();

  /* virtual */ void              draw                      ();
  
  /* virtual */ void              transform                 ( gpucast::gl::matrix4f const& m );

  /* virtual */ void              compute_nearfar           ();

  /* virtual */ void              recompile                 ();

  /* virtual */ void              resize                    ( int w, int h );

  /* virtual */ void              set_external_texture      ( std::shared_ptr<gpucast::gl::texture2d> const& texture );

  void                            register_cuda_resources   ();
  void                            unregister_cuda_resources ();

  virtual void                    register_cuda_structure   () = 0;
  virtual void                    unregister_cuda_structure () = 0;

  virtual void                    invoke_ray_casting_kernel ( renderconfig const& config ) = 0;
  virtual void                    create_data_structure     () = 0;

  virtual void                    update_gl_resources       ();
  virtual void                    update_gl_structure       () = 0;

  virtual void                    init_structure            () = 0;

  virtual void                    write                     ( std::ostream& os ) const;
  virtual void                    read                      ( std::istream& is );

protected : // auxilliary methods

  void                            _init_shader                ();

  void                            _init_gl_resources          ();
  
  void                            _update_matrixbuffer        ();

  void                            _draw_result                ();
  void                            _clean_textures             ();

  void                            _extract_faces              ( std::vector<face_ptr>& faces );

  bool                            _serialize                  ( std::string const& attribute_name );
                                                              
  void                            _create_proxy_geometry      ( beziervolume const& v, std::vector<deferred_surface_header_write>& deferred_jobs );
  void                            _build_paralleliped         ( beziervolume const& v, enum beziervolume::boundary_t face_type, deferred_surface_header_write& job );
                                                              
  void                            _verify_convexhull          ( std::vector<gpucast::math::point3d>& vertices,
                                                                std::vector<int>&          indices ) const;

protected : // attributes

  gpucast::gl::matrix4f                              _modelmatrix;

  gpucast::math::bbox3f                                 _bbox;
  bool                                        _gl_initialized;
  bool                                        _gl_updated;

  static unsigned const                       _empty_slots = 1;

  std::vector<gpucast::gl::vec4f>      _corners;

  std::vector<gpucast::gl::vec4u>      _surface_data;  
  //-----------------------------------------------------------------------------   
  // [attrib_min]          [bezierobject_id]           [inner/outer]  [uid]         
  // [attrib_max]          [surface_mesh_base_id]      [order_u]      [vid]         
  // [volume_buffer_id]    [adjacent_volume_id]        [order_v]      [wid]         
  // [attribute_data_id]   [adjacent_attribute_id]     [surface_type] [fixed_param] 
  //-----------------------------------------------------------------------------

  std::vector<gpucast::gl::vec4f>      _surface_points;
  //-----------------------------------------------------------------------------
  // [wx_00] [wx_01] ...
  // [wy_00] [wy_01] ...
  // [wz_00] [wz_01] ...
  // [ w_00] [ w_01] ...
  //-----------------------------------------------------------------------------

  std::vector<gpucast::gl::vec4f>      _volume_data; 
  //-----------------------------------------------------------------------------
  // [ volume_points_id  ] [ order_u ]      [ umin_local ] [ umax_local ] [ umin_global ] [ umax_global ] [ bbox_min ] [ bbox_max ]
  // [ uid               ] [ order_v ]      [ vmin_local ] [ vmax_local ] [ vmin_global ] [ vmax_global ] [ bbox_min ] [ bbox_max ] 
  // [ attribute_data_id ] [ order_w ]      [ wmin_local ] [ wmax_local ] [ wmin_global ] [ wmax_global ] [ bbox_min ] [ bbox_max ]
  // [ bbox_diameter     ] [ 0 ]            [ 0 ]          [ 0 ]          [ 0 ]           [ 0 ]           [ bbox_min ] [ bbox_max ]
  //-----------------------------------------------------------------------------

  std::vector<gpucast::gl::vec4f>      _volume_points;
  //-----------------------------------------------------------------------------
  // volume_points_id
  // [ wx_000 ]      [ wx_001 ] ...
  // [ wy_000 ]      [ wy_001 ] ...
  // [ wz_000 ]      [ wz_001 ] ...
  // [  w_000 ]      [  w_001 ] ...
  //-----------------------------------------------------------------------------

  std::vector<gpucast::gl::vec4f>      _attribute_data; // attribute data buffers
  // attribute_data_id
  //-----------------------------------------------------------------------------
  // [ min ]       
  // [ max ]       
  // [ attribute_point_id ]  
  // [ empty ]
  //-----------------------------------------------------------------------------

  std::vector<gpucast::gl::vec2f>      _attribute_points; // attribute buffers
  // attribute_point_id
  //-----------------------------------------------------------------------------
  // [attrib0_000] ...
  // [attrib1_000] ...
  //-----------------------------------------------------------------------------

  std::shared_ptr<gpucast::gl::arraybuffer>        _volume_data_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _volume_points_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _surface_data_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _surface_points_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _attribute_data_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _attribute_points_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _matrix_arraybuffer;

  // GL input resources
  std::shared_ptr<gpucast::gl::program>            _raygeneration_program;
  std::shared_ptr<gpucast::gl::cube>               _raygeneration_geometry;
  std::shared_ptr<gpucast::gl::framebufferobject>  _raygeneration_fbo;
  std::shared_ptr<gpucast::gl::texture2d>          _raygeneration_color;
  std::shared_ptr<gpucast::gl::renderbuffer>       _raygeneration_depth;

  std::shared_ptr<gpucast::gl::framebufferobject>  _output_fbo;
  std::shared_ptr<gpucast::gl::texture2d>          _color_texture;
  std::shared_ptr<gpucast::gl::texture2d>          _depth_texture;
  std::shared_ptr<gpucast::gl::sampler>            _nearest_interpolation;

  // CUDA resources
  bool                                        _cuda_resources_mapped;
  cudaGraphicsResource*                       _cuda_input_color_depth;
  cudaGraphicsResource*                       _cuda_external_color_depth;

  cudaGraphicsResource*                       _cuda_surface_data_buffer;
  cudaGraphicsResource*                       _cuda_surface_points_buffer;
  cudaGraphicsResource*                       _cuda_volume_data_buffer;
  cudaGraphicsResource*                       _cuda_volume_points_buffer;
  cudaGraphicsResource*                       _cuda_attribute_data_buffer;
  cudaGraphicsResource*                       _cuda_attribute_points_buffer;
  cudaGraphicsResource*                       _cuda_matrixbuffer;

  cudaGraphicsResource*                       _cuda_output_color;
  cudaGraphicsResource*                       _cuda_output_depth;

  // final gl mapping
  std::shared_ptr<gpucast::gl::program>            _map_quad_program;
  std::shared_ptr<gpucast::gl::plane>              _map_quad;
};

} // namespace gpucast

#endif // GPUCAST_ISOURFACE_RENDERER_STRUCTURE_BASED_HPP

