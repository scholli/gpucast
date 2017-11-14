/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bezierobject_renderer.hpp
*  project    : gpucast::gl
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_BEZIEROBJECT_RENDERER_HPP
#define GPUCAST_GL_BEZIEROBJECT_RENDERER_HPP

#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/singleton.hpp>

#include <gpucast/gl/util/material.hpp>
#include <gpucast/gl/util/hullvertexmap.hpp>
#include <gpucast/gl/util/resource_factory.hpp>

#include <gpucast/gl/glpp.hpp>

#include <gpucast/gl/primitives/bezierobject.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/atomicbuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/transformfeedback.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/shaderstoragebuffer.hpp>

#include <gpucast/gl/primitives/plane.hpp>



namespace gpucast { 

  namespace gl {

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL bezierobject_renderer : public singleton<bezierobject_renderer>
{
public: // enums, typedefs
  
  static const unsigned MAX_XFB_BUFFER_SIZE_IN_BYTES                 = 1024000000; // reserve GB transform feedback buffer
  static const unsigned MAX_FEEDBACK_BUFFER_INDICES                  = 1024000;
  static const unsigned GPUCAST_HULLVERTEXMAP_SSBO_BINDING           = 1;
  static const unsigned GPUCAST_ATTRIBUTE_SSBO_BINDING               = 2;
  static const unsigned GPUCAST_ATOMIC_COUNTER_BINDING               = 3;
  static const unsigned GPUCAST_FEEDBACK_BUFFER_BINDING              = 4;

  static const unsigned GPUCAST_ABUFFER_MAX_FRAGMENTS                = 10000000; // 10M fragments 
  static const unsigned GPUCAST_ABUFFER_ATOMIC_BUFFER_BINDING        = 5;
  static const unsigned GPUCAST_ABUFFER_FRAGMENT_LIST_BUFFER_BINDING = 6;
  static const unsigned GPUCAST_ABUFFER_FRAGMENT_DATA_BUFFER_BINDING = 7;

  static const unsigned GPUCAST_OBJECT_UBO_BINDINGPOINT = 0;
  static const unsigned GPUCAST_CAMERA_UBO_BINDINGPOINT = 1;

  static const unsigned GPUCAST_ANTI_ALIASING_MODE                   = bezierobject::anti_aliasing_mode::disabled;

  struct debug_counter {
    unsigned triangles;
    unsigned fragments;
    unsigned culled_triangles;
    unsigned trimmed_fragments;
  };

  struct matrix_uniform_buffer_layout
  {
    gpucast::math::matrix4f gpucast_model_matrix; // used
    gpucast::math::matrix4f gpucast_model_inverse_matrix;

    gpucast::math::matrix4f gpucast_model_view_matrix; // used
    gpucast::math::matrix4f gpucast_model_view_inverse_matrix;

    gpucast::math::matrix4f gpucast_model_view_projection_matrix; // used
    gpucast::math::matrix4f gpucast_model_view_projection_inverse_matrix;

    gpucast::math::matrix4f gpucast_normal_matrix;  // used

    gpucast::math::matrix4f gpucast_view_matrix;
    gpucast::math::matrix4f gpucast_view_inverse_matrix; // used

    gpucast::math::matrix4f gpucast_projection_matrix;
    gpucast::math::matrix4f gpucast_projection_inverse_matrix;

    gpucast::math::matrix4f gpucast_projection_view_matrix;
    gpucast::math::matrix4f gpucast_projection_view_inverse_matrix;

    gpucast::math::vec2i    gpucast_resolution;

    float                   gpucast_clip_near;
    float                   gpucast_clip_far;
  };


public: // c'tor / d'tor

  bezierobject_renderer();
  ~bezierobject_renderer();

public: // methods

  int                           next_texunit();

  void                          set_custom_transform_feedback_size(std::size_t bytes);
  void                          set_abuffer_size(std::size_t max_fragments);

  std::shared_ptr<program> const& get_raycasting_program() const;
  std::shared_ptr<program> const& get_pretesselation_program() const;
  std::shared_ptr<program> const& get_tesselation_program() const;

  void           attach_custom_textures(std::shared_ptr<texture2d> const& color_texture, std::shared_ptr<texture2d> const& depth_texture);

  void           create_fbo();
  void           create_multisample_fbo();
  void           create_gbuffer();

  void           begin_draw();
  void           end_draw();

  void           set_nearfar(float near, float far);
  void           set_resolution(unsigned width, unsigned height);
  gpucast::math::vec2i const& get_resolution() const;

  void           set_background(gpucast::math::vec3f const& color);
  void           add_search_path(std::string const& path);

  void           spheremapping(bool);
  void           spheremap(std::string const& filepath);
  void           diffusemapping(bool);
  void           diffusemap(std::string const& filepath);

  void           view_setup(gpucast::math::matrix4f const& view, 
                            gpucast::math::matrix4f const& model, 
                            gpucast::math::matrix4f const& projection);

  void           current_viewmatrix(gpucast::math::matrix4f const& m);
  void           current_modelmatrix(gpucast::math::matrix4f const& m);
  void           current_projectionmatrix(gpucast::math::matrix4f const& m);

  bool           inside_frustum(bezierobject const&) const;
  void           recompile();
                 
  void           begin_program(std::shared_ptr<program> const& p);
  void           end_program(std::shared_ptr<program> const& p);             
  void           apply_uniforms(std::shared_ptr<program> const& p);

  void           antialiasing(bezierobject::anti_aliasing_mode m);
  bezierobject::anti_aliasing_mode antialiasing() const;

  void           enable_conservative_rasterization(bool);
  bool           enable_conservative_rasterization() const;

  void           enable_holefilling(bool);
  bool           enable_holefilling() const;

  void           enable_counting(bool);
  bool           enable_counting() const;

  void           enable_triangular_tesselation(bool);
  bool           enable_triangular_tesselation() const;

  debug_counter  get_debug_count() const;
  std::vector<unsigned> get_fragment_estimate() const;

private : // methods

  void _abuffer_clear();

  void _init_raycasting_program();
  void _init_pretesselation_program();
  void _init_tesselation_program();
  void _init_resolve_program();
  void _init_hullvertexmap();
  void _init_prefilter(unsigned prefilter_resolution = 128);
  void _init_transform_feedback();

private: // attributes

  std::size_t                           _abuffer_max_fragment = GPUCAST_ABUFFER_MAX_FRAGMENTS;
  std::size_t                           _xfb_budget_bytes     = MAX_XFB_BUFFER_SIZE_IN_BYTES;


  bool                                  _conservative_rasterization = false;
  bool                                  _enable_holefilling = false;
  bool                                  _enable_count = false;
  bool                                  _enable_triangular_tesselation = false;

  bezierobject::anti_aliasing_mode      _antialiasing;
                      
  std::shared_ptr<shaderstoragebuffer>  _hullvertexmap;
  std::set<std::string>                 _pathlist;
  gpucast::math::vec3f                  _background;
                                        
  int                                   _texunit = 0;

  // surface_renderer global ressources
  gpucast::gl::resource_factory         _program_factory;

  std::shared_ptr<program>              _raycasting_program;
  std::shared_ptr<program>              _pretesselation_program;
  std::shared_ptr<program>              _tesselation_program;
  std::shared_ptr<program>              _resolve_program;
                                        
  std::shared_ptr<atomicbuffer>         _counter;
  std::shared_ptr<shaderstoragebuffer>  _feedbackbuffer;

  bool                                  _enable_spheremap = false;
  bool                                  _enable_diffusemap = false;
  std::shared_ptr<texture2d>            _spheremap;
  std::shared_ptr<texture2d>            _diffusemap;
                                        
  std::shared_ptr<texture2d>            _prefilter_texture;
  std::shared_ptr<sampler>              _linear_sampler;
  std::shared_ptr<sampler>              _nearest_sampler;

  // off-screen render targets
  std::shared_ptr<gpucast::gl::framebufferobject>   _gbuffer;
  std::shared_ptr<gpucast::gl::texture2d>           _gbuffer_depthattachment;
  std::shared_ptr<gpucast::gl::texture2d>           _gbuffer_colorattachment;
  std::shared_ptr<gpucast::gl::plane>               _fullscreen_quad;

  std::shared_ptr<gpucast::gl::framebufferobject>   _fbo;
  std::shared_ptr<gpucast::gl::texture2d>           _depthattachment;
  std::shared_ptr<gpucast::gl::texture2d>           _colorattachment;
                                                    
  std::shared_ptr<gpucast::gl::framebufferobject>   _fbo_multisample;
  std::shared_ptr<gpucast::gl::renderbuffer>        _colorattachment_multisample;
  std::shared_ptr<gpucast::gl::renderbuffer>        _depthattachment_multisample;

  std::shared_ptr<gpucast::gl::atomicbuffer>        _abuffer_atomic_buffer;
  std::shared_ptr<gpucast::gl::shaderstoragebuffer> _abuffer_fragment_list;
  std::shared_ptr<gpucast::gl::shaderstoragebuffer> _abuffer_fragment_data;

  matrix_uniform_buffer_layout                      _camera_ubo_data;
  std::shared_ptr<gpucast::gl::uniformbuffer>       _camera_ubo;

  struct {
    GLint                                             _polygonmode;
    bool                                              _conservative_rasterization_enabled;
  } _glstate_backup;
  
};


} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_BEZIEROBJECT_HPP

