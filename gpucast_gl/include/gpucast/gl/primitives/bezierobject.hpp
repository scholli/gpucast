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
public : 

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
    FILL_SOLID = 0x00,
    FILL_WIREFRAME,
    FILL_POINT,
    FILL_MODE_COUNT
  }; 

  enum render_mode {
    raycasting = 0x00,
    tesselation
  };

  bezierobject (gpucast::beziersurfaceobject const&);

  bezierobject(bezierobject const&) = delete;
  bezierobject& operator=(gpucast::beziersurfaceobject const&) = delete;

  gpucast::beziersurfaceobject const& object() const;

  // draw methods
  void                draw(render_mode mode = raycasting);

  // configuration
  void                max_newton_iterations ( unsigned n );
  unsigned            max_newton_iterations() const;

  void                newton_epsilon ( float epsilon );
  float               newton_epsilon() const;

  void                culling ( bool enable );
  bool                culling () const;

  void                enable_raycasting(bool enable);
  bool                enable_raycasting() const;

  beziersurfaceobject::trim_approach_t trimming() const;
  void                trimming(beziersurfaceobject::trim_approach_t type);

  void                antialiasing(enum anti_aliasing_mode);
  anti_aliasing_mode  antialiasing() const;

  void                set_material(material const& m);
  material const&     get_material() const;

private :

  void _draw_by_raycasting();
  void _draw_by_tesselation();

  void _apply_uniforms(program const& p);

  void _upload();
  void _upload_trimming_buffers();
  void _upload_raycasting_buffers();
  void _upload_tesselation_buffers();

  // ray casting parameters
  unsigned                              _iterations    = 6;
  float                                 _epsilon       = 0.001f;
  bool                                  _culling       = true;
  bool                                  _raycasting    = true;
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
  fill_mode                             _fill_mode;

  gpucast::gl::vertexarrayobject        _tesselation_vertex_array;

  gpucast::gl::arraybuffer              _tesselation_vertex_buffer;
  gpucast::gl::arraybuffer              _tesselation_index_buffer;
  gpucast::gl::arraybuffer              _tesselation_hullvertexmap;
  gpucast::gl::arraybuffer              _tesselation_attribute_buffer;

  gpucast::gl::texturebuffer            _tesselation_parametric_texture_buffer;
  gpucast::gl::texturebuffer            _tesselation_domain_texture_buffer;
  gpucast::gl::texturebuffer            _tesselation_attribute_texture_buffer;

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
class GPUCAST_GL bezierobject_renderer
{
public: // enums, typedefs
  
  static const unsigned MAX_XFB_BUFFER_SIZE_IN_BYTES = 1024000000; // reserve GB transform feedback buffer

private: // c'tor / d'tor

  bezierobject_renderer();
  bezierobject_renderer(bezierobject_renderer const& other) = delete;
  bezierobject_renderer& operator=(bezierobject_renderer const& other) = delete;

public:

  ~bezierobject_renderer();
  static bezierobject_renderer& instance();

public: // methods

  int            next_texunit();

  std::shared_ptr<program> const& get_raycasting_program() const;
  std::shared_ptr<program> const& get_pretesselation_program() const;
  std::shared_ptr<program> const& get_tesselation_program() const;

  void           set_nearfar(float near, float far);
  void           set_background(gpucast::math::vec3f const& color);
  void           add_search_path(std::string const& path);

  void           spheremap(std::string const& filepath);
                 
  void           diffusemap(std::string const& filepath);

  void           modelviewmatrix(gpucast::math::matrix4f const& mv);
  void           projectionmatrix(gpucast::math::matrix4f const& projection);

  void           recompile();
                 
  void           begin_program(std::shared_ptr<program> const& p);
  void           end_program(std::shared_ptr<program> const& p);             
  void           apply_uniforms(std::shared_ptr<program> const& p);

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
                              
  gpucast::math::matrix4f       _modelviewmatrix;
  gpucast::math::matrix4f       _modelviewmatrixinverse;
  gpucast::math::matrix4f       _projectionmatrix;
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

  std::shared_ptr<texture2d>    _spheremap;
  std::shared_ptr<texture2d>    _diffusemap;
  std::shared_ptr<texture2d>    _prefilter_texture;

  struct transform_feedback_buffer {
    std::shared_ptr<gpucast::gl::transform_feedback> _transform_feedback;
    std::shared_ptr<gpucast::gl::vertexarrayobject>  _transform_feedback_vao;
    std::shared_ptr<gpucast::gl::arraybuffer>        _transform_feedback_vbo;
  };
};


} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_BEZIEROBJECT_HPP
