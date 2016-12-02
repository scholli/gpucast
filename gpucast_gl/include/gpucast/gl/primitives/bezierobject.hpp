/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bezierobject.hpp
*  project    : gpucast::gl
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
    static const float    tesselation_max_geometric_error; // default set to 0.0001f = 0.1 mm
  };

  enum anti_aliasing_mode {
    disabled = 0x00,
    prefiltered_edge_estimation = 0x01,
    multisampling2x2 = 0x02,
    multisampling3x3 = 0x03,
    multisampling4x4 = 0x04,
    multisampling8x8 = 0x05,
    msaa             = 0x06,
    fxaa             = 0x07,
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

  void                tesselation_max_geometric_error(float epsilon);
  float               tesselation_max_geometric_error() const;

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
  void _upload_controlpoint_buffer();
  void _upload_raycasting_buffers();
  void _upload_tesselation_buffers();

  // ray casting parameters
  unsigned                              _raycasting_max_iterations      = default_render_configuration::raycasting_max_iterations;
  unsigned                              _trimming_max_bisections        = default_render_configuration::trimming_max_bisections;
  float                                 _raycasting_error_tolerance     = default_render_configuration::raycasting_error_tolerance;
  float                                 _trimming_error_tolerance       = default_render_configuration::trimming_error_tolerance;
  float                                 _tesselation_max_pixel_error    = default_render_configuration::tesselation_max_pixel_error;
  float                                 _tesselation_max_pretesselation = default_render_configuration::tesselation_max_pretesselation;
  float                                 _tesselation_max_geometric_error = default_render_configuration::tesselation_max_geometric_error;

  bool                                  _culling                 = false;
  bool                                  _raycasting              = true;

  render_mode                           _rendermode    = raycasting;
  anti_aliasing_mode                    _antialiasing  = disabled;
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

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_BEZIEROBJECT_HPP
