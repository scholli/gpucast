/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : plane.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_BEZIEROBJECT_HPP
#define GPUCAST_GL_BEZIEROBJECT_HPP

#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/singleton.hpp>

#include <gpucast/gl/util/material.hpp>
#include <gpucast/gl/util/hullvertexmap.hpp>

#include <gpucast/gl/glpp.hpp>
\
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/atomicbuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/transformfeedback.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/shaderstoragebuffer.hpp>



namespace gpucast { namespace gl {

  class arraybuffer;

class GPUCAST_GL bezierobject 
{
public:

  struct GPUCAST_GL default_render_configuration {
    static const unsigned raycasting_max_iterations = 6;
    static const float    raycasting_error_tolerance;     // default set to 0.001f
    static const unsigned trimming_max_bisections = 16;
    static const float    trimming_error_tolerance;       // default set to 0.001f
    static const float    tesselation_max_pixel_error;    // default set to 4.0 pixel
    static const float    tesselation_max_pretesselation; // default set to 64.0f
  };

  enum anti_aliasing_mode {
    no_anti_aliasing,
    prefiltered_edge_estimation,
    supersampling2x2,
    supersampling3x3,
    supersampling4x4,
    supersampling8x8,
  };

  enum fill_mode
  {
    solid = 0x00,
    wireframe,
    points
  }; 

  enum render_mode {
    raycasting    = 0x00,
    tesselation   = 0x01,
    shadow        = 0x02,
    shadow_lowres = 0x03
  };

  bezierobject (gpucast::beziersurfaceobject const&);

  bezierobject(bezierobject const&) = delete;
  bezierobject& operator=(gpucast::beziersurfaceobject const&) = delete;

  gpucast::beziersurfaceobject const& object() const;
  void                init(unsigned subdiv_u, unsigned subdiv_v, unsigned preclass_resolution);

  // draw methods
  void                draw();

  // configuration
  void                raycasting_max_iterations ( unsigned n );
  unsigned            raycasting_max_iterations() const;

  void                trimming_max_bisections(unsigned n);
  unsigned            trimming_max_bisections() const;

  void                raycasting_error_tolerance ( float epsilon );
  float               raycasting_error_tolerance() const;

  void                trimming_error_tolerance(float epsilon);
  float               trimming_error_tolerance() const;

  void                tesselation_max_pixel_error(float epsilon);
  float               tesselation_max_pixel_error() const;

  void                tesselation_max_pretesselation(float epsilon);
  float               tesselation_max_pretesselation() const;

  void                culling ( bool enable );
  bool                culling () const;

  void                enable_raycasting(bool enable);
  bool                enable_raycasting() const;

  void                rendermode(render_mode mode);
  render_mode         rendermode() const;

  beziersurfaceobject::trim_approach_t trimming() const;
  void                trimming(beziersurfaceobject::trim_approach_t type);

  void                antialiasing(enum anti_aliasing_mode);
  anti_aliasing_mode  antialiasing() const;

  void                fillmode(fill_mode mode);
  fill_mode           fillmode() const;

  void                set_material(material const& m);
  material const&     get_material() const;

private :

  void _draw_by_raycasting();
  void _draw_by_tesselation();

  void _apply_uniforms(program const& p, render_mode mode);

  void _upload();
  void _upload_trimming_buffers();
  void _upload_raycasting_buffers();
  void _upload_tesselation_buffers();

  // ray casting parameters
  unsigned                              _raycasting_max_iterations      = default_render_configuration::raycasting_max_iterations;
  unsigned                              _trimming_max_bisections        = default_render_configuration::trimming_max_bisections;
  float                                 _raycasting_error_tolerance     = default_render_configuration::raycasting_error_tolerance;
  float                                 _trimming_error_tolerance       = default_render_configuration::trimming_error_tolerance;
  float                                 _tesselation_max_pixel_error    = default_render_configuration::tesselation_max_pixel_error;
  float                                 _tesselation_max_pretesselation = default_render_configuration::tesselation_max_pretesselation;

  bool                                  _culling                 = false;
  bool                                  _raycasting              = true;

  render_mode                           _rendermode    = tesselation;
  anti_aliasing_mode                    _antialiasing  = no_anti_aliasing;
  beziersurfaceobject::trim_approach_t  _trimming      = beziersurfaceobject::contour_kd_partition;
    
  beziersurfaceobject                   _object;

  // object properties
  material                              _material;

  // gpu ressources : ray casting 
  std::size_t                           _size;

  gpucast::gl::elementarraybuffer       _chull_indexarray;

  gpucast::gl::arraybuffer              _chull_attribarray0;
  gpucast::gl::arraybuffer              _chull_attribarray1;
  gpucast::gl::arraybuffer              _chull_attribarray2;
  gpucast::gl::arraybuffer              _chull_attribarray3;

  gpucast::gl::vertexarrayobject        _chull_vao;

  gpucast::gl::texturebuffer            _controlpoints;
  gpucast::gl::texturebuffer            _obbs;
             

  // gpu ressources : adaptive tesselation 
  fill_mode                             _fill_mode = solid;
  int                                   _tesselation_vertex_count;

  gpucast::gl::vertexarrayobject        _tesselation_vertex_array;

  gpucast::gl::arraybuffer              _tesselation_vertex_buffer;
  gpucast::gl::elementarraybuffer       _tesselation_index_buffer;
  gpucast::gl::arraybuffer              _tesselation_hullvertexmap;
  gpucast::gl::shaderstoragebuffer      _tesselation_attribute_buffer;

  gpucast::gl::texturebuffer            _tesselation_parametric_texture_buffer;

  // gpu ressources trimming
  gpucast::gl::texturebuffer            _cmb_partition;
  gpucast::gl::texturebuffer            _cmb_contourlist;
  gpucast::gl::texturebuffer            _cmb_curvelist;
  gpucast::gl::texturebuffer            _cmb_curvedata;
  gpucast::gl::texturebuffer            _cmb_pointdata;
  gpucast::gl::texturebuffer            _cmb_preclassification;

  gpucast::gl::texturebuffer            _db_partition;
  gpucast::gl::texturebuffer            _db_celldata;
  gpucast::gl::texturebuffer            _db_curvelist;
  gpucast::gl::texturebuffer            _db_curvedata;
  gpucast::gl::texturebuffer            _db_preclassification;

  gpucast::gl::texturebuffer            _kd_partition;
  gpucast::gl::texturebuffer            _kd_contourlist;
  gpucast::gl::texturebuffer            _kd_curvelist;
  gpucast::gl::texturebuffer            _kd_curvedata;
  gpucast::gl::texturebuffer            _kd_pointdata;
  gpucast::gl::texturebuffer            _kd_preclassification;

  gpucast::gl::shaderstoragebuffer      _loop_list_loops;
  gpucast::gl::shaderstoragebuffer      _loop_list_contours;
  gpucast::gl::shaderstoragebuffer      _loop_list_curves;
  gpucast::gl::shaderstoragebuffer      _loop_list_points;
  gpucast::gl::texturebuffer            _loop_list_preclassification;
};

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL bezierobject_renderer : public singleton<bezierobject_renderer>
{
public: // enums, typedefs
  
  static const unsigned MAX_XFB_BUFFER_SIZE_IN_BYTES = 1024000000; // reserve GB transform feedback buffer
  static const unsigned GPUCAST_HULLVERTEXMAP_SSBO_BINDING = 1;
  static const unsigned GPUCAST_ATTRIBUTE_SSBO_BINDING = 2;
  static const unsigned GPUCAST_ATOMIC_COUNTER_BINDING = 3;
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

  void           enable_counting(bool);
  bool           enable_counting() const;

  unsigned       get_fragment_count() const;
  unsigned       get_triangle_count() const;
  void           reset_count() const;

private : // methods

  void _init_raycasting_program();
  void _init_pretesselation_program();
  void _init_tesselation_program();
  void _init_hullvertexmap();
  void _init_prefilter(unsigned prefilter_resolution = 128);
  void _init_transform_feedback();

private: // attributes

  float                         _nearplane;
  float                         _farplane;
  bool                          _enable_count = false;

  gpucast::math::vec2i          _resolution;

  gpucast::math::matrix4f       _modelmatrix;
  gpucast::math::matrix4f       _modelmatrixinverse;

  gpucast::math::matrix4f       _viewmatrix;
  gpucast::math::matrix4f       _viewmatrixinverse;

  gpucast::math::matrix4f       _modelviewmatrix;
  gpucast::math::matrix4f       _modelviewmatrixinverse;

  gpucast::math::matrix4f       _projectionmatrix;
  gpucast::math::matrix4f       _projectionmatrixinverse;

  gpucast::math::matrix4f       _normalmatrix;
  gpucast::math::matrix4f       _modelviewprojectionmatrix;
  gpucast::math::matrix4f       _modelviewprojectionmatrixinverse;
                              
  std::shared_ptr<shaderstoragebuffer> _hullvertexmap;
  std::set<std::string>         _pathlist;
  gpucast::math::vec3f          _background;

  int                           _texunit = 0;

  // surface_renderer global ressources
  std::shared_ptr<program>      _raycasting_program;
  std::shared_ptr<program>      _pretesselation_program;
  std::shared_ptr<program>      _tesselation_program;

  std::shared_ptr<atomicbuffer> _counter;
  std::shared_ptr<texture2d>    _spheremap;
  std::shared_ptr<texture2d>    _diffusemap;
  std::shared_ptr<texture2d>    _prefilter_texture;
};


} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_BEZIEROBJECT_HPP
