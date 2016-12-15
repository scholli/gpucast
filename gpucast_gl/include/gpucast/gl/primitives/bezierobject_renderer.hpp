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
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/transformfeedback.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/shaderstoragebuffer.hpp>



namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL bezierobject_renderer : public singleton<bezierobject_renderer>
{
public: // enums, typedefs
  
  static const unsigned MAX_XFB_BUFFER_SIZE_IN_BYTES        = 1024000000; // reserve GB transform feedback buffer
  static const unsigned MAX_FEEDBACK_BUFFER_INDICES         = 1024;
  static const unsigned GPUCAST_HULLVERTEXMAP_SSBO_BINDING  = 1;
  static const unsigned GPUCAST_ATTRIBUTE_SSBO_BINDING      = 2;
  static const unsigned GPUCAST_ATOMIC_COUNTER_BINDING      = 3;
  static const unsigned GPUCAST_FEEDBACK_BUFFER_BINDING     = 4;
  static const unsigned GPUCAST_ANTI_ALIASING_MODE          = bezierobject::anti_aliasing_mode::disabled;
  
  struct debug_counter {
    unsigned fragments;
    unsigned triangles;
    unsigned culled_triangles;
    unsigned trimmed_fragments;
  };

public: // c'tor / d'tor

  bezierobject_renderer();
  ~bezierobject_renderer();

public: // methods

  int            next_texunit();

  std::shared_ptr<program> const& get_raycasting_program() const;
  std::shared_ptr<program> const& get_pretesselation_program() const;
  std::shared_ptr<program> const& get_tesselation_program() const;

  void           set_nearfar(float near, float far);
  void           set_resolution(unsigned width, unsigned height);
  void           set_background(gpucast::math::vec3f const& color);
  void           add_search_path(std::string const& path);

  void           spheremap(std::string const& filepath);
                 
  void           diffusemap(std::string const& filepath);

  void           current_viewmatrix(gpucast::math::matrix4f const& m);
  void           current_modelmatrix(gpucast::math::matrix4f const& m);
  void           current_projectionmatrix(gpucast::math::matrix4f const& m);

  void           recompile();
                 
  void           begin_program(std::shared_ptr<program> const& p);
  void           end_program(std::shared_ptr<program> const& p);             
  void           apply_uniforms(std::shared_ptr<program> const& p);

  void           antialiasing(bezierobject::anti_aliasing_mode m);
  bezierobject::anti_aliasing_mode antialiasing() const;

  void           enable_counting(bool);
  bool           enable_counting() const;

  void           enable_triangular_tesselation(bool);
  bool           enable_triangular_tesselation() const;

  debug_counter  get_debug_count() const;
  std::vector<unsigned> get_fragment_estimate() const;

private : // methods

  void _init_raycasting_program();
  void _init_pretesselation_program();
  void _init_tesselation_program();
  void _init_hullvertexmap();
  void _init_prefilter(unsigned prefilter_resolution = 128);
  void _init_transform_feedback();

private: // attributes

  float                            _nearplane;
  float                            _farplane;
  bool                             _enable_count = false;
  bool                             _enable_triangular_tesselation = false;
  bezierobject::anti_aliasing_mode _antialiasing;

  gpucast::math::vec2i             _resolution;
                                   
  gpucast::math::matrix4f          _modelmatrix;
  gpucast::math::matrix4f          _modelmatrixinverse;
                                   
  gpucast::math::matrix4f          _viewmatrix;
  gpucast::math::matrix4f          _viewmatrixinverse;
                                   
  gpucast::math::matrix4f          _modelviewmatrix;
  gpucast::math::matrix4f          _modelviewmatrixinverse;
                                   
  gpucast::math::matrix4f          _projectionmatrix;
  gpucast::math::matrix4f          _projectionmatrixinverse;
                                   
  gpucast::math::matrix4f          _normalmatrix;
  gpucast::math::matrix4f          _modelviewprojectionmatrix;
  gpucast::math::matrix4f          _modelviewprojectionmatrixinverse;
                              
  std::shared_ptr<shaderstoragebuffer> _hullvertexmap;
  std::set<std::string>            _pathlist;
  gpucast::math::vec3f             _background;
                                   
  int                              _texunit = 0;

  // surface_renderer global ressources
  gpucast::gl::resource_factory    _program_factory;
  std::shared_ptr<program>         _raycasting_program;
  std::shared_ptr<program>         _pretesselation_program;
  std::shared_ptr<program>         _tesselation_program;
                                   
  std::shared_ptr<atomicbuffer>    _counter;
  std::shared_ptr<shaderstoragebuffer> _feedbackbuffer;

  std::shared_ptr<texture2d>       _spheremap;
  std::shared_ptr<texture2d>       _diffusemap;
                                   
  std::shared_ptr<texture2d>       _prefilter_texture;
  std::shared_ptr<sampler>         _prefilter_sampler;
};


} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_BEZIEROBJECT_HPP