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

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>


namespace gpucast { namespace gl {

  class arraybuffer;

class GPUCAST_GL bezierobject 
{
public : 

  bezierobject (gpucast::beziersurfaceobject const&);

  bezierobject(bezierobject const&) = delete;
  bezierobject& operator=(gpucast::beziersurfaceobject const&) = delete;

  ~bezierobject();

  // draw methods
  void            draw();

  // configuration
  void            max_newton_iterations ( unsigned n );
  unsigned        max_newton_iterations() const;

  void            newton_epsilon ( float epsilon );
  float           newton_epsilon() const;

  void            culling ( bool enable );
  bool            culling () const;

  void            trimming(bool enable);
  bool            trimming() const;

  void            raycasting(bool enable);
  bool            raycasting() const;

  beziersurfaceobject::trim_approach_t trim_approach() const;
  void            trim_approach(beziersurfaceobject::trim_approach_t type);

  void            set_material(material const& m);
  material const& get_material() const;

private :

  void _apply_uniforms(program const& p);

  void _upload(gpucast::beziersurfaceobject const& b);

  // render parameters
  unsigned                              _iterations    = 6;
  float                                 _epsilon       = 0.001f;
  bool                                  _culling       = true;
  bool                                  _trimming      = true;
  bool                                  _raycasting    = true;
  beziersurfaceobject::trim_approach_t  _trim_approach = beziersurfaceobject::double_binary;
  
  // object properties
  material                              _material;

  // ressources
  std::size_t                           _size;

  gpucast::gl::arraybuffer              _attribarray0;
  gpucast::gl::arraybuffer              _attribarray1;
  gpucast::gl::arraybuffer              _attribarray2;
  gpucast::gl::arraybuffer              _attribarray3;

  gpucast::gl::elementarraybuffer       _indexarray;

  gpucast::gl::texturebuffer            _controlpoints;
                                        
  gpucast::gl::texturebuffer            _cmb_partition;
  gpucast::gl::texturebuffer            _cmb_contourlist;
  gpucast::gl::texturebuffer            _cmb_curvelist;
  gpucast::gl::texturebuffer            _cmb_curvedata;
  gpucast::gl::texturebuffer            _cmb_pointdata;

  gpucast::gl::texturebuffer            _db_partition;
  gpucast::gl::texturebuffer            _db_celldata;
  gpucast::gl::texturebuffer            _db_curvelist;
  gpucast::gl::texturebuffer            _db_curvedata;

  gpucast::gl::vertexarrayobject        _vao;
};

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL bezierobject_renderer
{
public: // enums, typedefs
  
private: // c'tor / d'tor

  bezierobject_renderer();
  bezierobject_renderer(bezierobject_renderer const& other) = delete;
  bezierobject_renderer& operator=(bezierobject_renderer const& other) = delete;

public:

  ~bezierobject_renderer();
  static bezierobject_renderer& instance();

public: // methods

  int            next_texunit();
  program const& get_program() const;

  void           set_nearfar(float near, float far);
  void           set_background(gpucast::math::vec3f const& color);
  void           add_search_path(std::string const& path);

  void           spheremap(std::string const& filepath);
                 
  void           diffusemap(std::string const& filepath);
                 
  void           cubemap(std::string const& positive_x,
                         std::string const& negative_x,
                         std::string const& positive_y,
                         std::string const& negative_y,
                         std::string const& positive_z,
                         std::string const& negative_z);

  void           modelviewmatrix(gpucast::math::matrix4f const& mv);
  void           projectionmatrix(gpucast::math::matrix4f const& projection);

  void           recompile();
                 
  void           bind();
  void           unbind();
                 
  void           apply_uniforms();

private : // methods

  void _init_program();

private: // attributes

  float                         _nearplane;
  float                         _farplane;
                              
  gpucast::math::matrix4f       _modelviewmatrix;
  gpucast::math::matrix4f       _modelviewmatrixinverse;
  gpucast::math::matrix4f       _projectionmatrix;
  gpucast::math::matrix4f       _normalmatrix;
  gpucast::math::matrix4f       _modelviewprojectionmatrix;
  gpucast::math::matrix4f       _modelviewprojectionmatrixinverse;
                              
  std::set<std::string>         _pathlist;
  gpucast::math::vec3f          _background;

  int                           _texunit = 0;

  // surface_renderer global ressources
  std::shared_ptr<program>    _program;
  std::shared_ptr<texture2d>  _spheremap;
  std::shared_ptr<texture2d>  _diffusemap;
};


} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_BEZIEROBJECT_HPP
