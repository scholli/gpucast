/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : glwidget.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef NURBSVIEW_GLWIDGET_HPP
#define NURBSVIEW_GLWIDGET_HPP

#if WIN32
  #pragma warning(disable: 4127) // Qt conditional expression is constant
  #pragma warning(disable: 4245) // CL warnings
#endif

// system includes
#include <GL/glew.h>

#if WIN32
  #include <QtOpenGL/QGLWidget>
#else
  #include <qt4/QtOpenGL/QGLWidget>
#endif

#include <vector>

#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/primitives/plane.hpp>
    
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/math/vec3.hpp>

#include <gpucast/math/axis_aligned_boundingbox.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/surface_renderer_gl.hpp>



class glwidget : public QGLWidget
{
   Q_OBJECT        // must include this if you use Qt signals/slots

public : 

  glwidget                                              ( int argc, char** argv, QGLFormat const& format, QWidget *parent = 0 );
  ~glwidget                                             ();

  void                    open                          ( std::list<std::string> const& );
  void                    add                           ( std::list<std::string> const& );

public Q_SLOTS : 

  void                    recompile                     ();
  void                    load_spheremap                ( );
  void                    load_diffusemap               ( );
  void                    spheremapping                 ( int ); 
  void                    diffusemapping                ( int );
  void                    fxaa                          ( int ); 
  void                    vsync                         ( int );
  void                    ambient_occlusion             ( int );

protected:

  /* virtual */ void      initializeGL                  ();
  /* virtual */ void      resizeGL                      ( int w, int h );
  /* virtual */ void      paintGL                       ();

  /* virtual */ void      mousePressEvent               ( QMouseEvent* event );
  /* virtual */ void      mouseReleaseEvent             ( QMouseEvent* event );
  /* virtual */ void      mouseMoveEvent                ( QMouseEvent* event );

  /* virtual */ void      keyPressEvent                 ( QKeyEvent* event);
  /* virtual */ void      keyReleaseEvent               ( QKeyEvent* event);

private : // helper methods

  void                    _init                         ();
  void                    _init_data                    ();
  void                    _create_shader                ();
  void                    _print_contextinfo            ();

  void                    _generate_ao_sampletexture    ();
  void                    _generate_random_texture      ();

  void                    _openfile                     ( std::string const& file, gpucast::math::axis_aligned_boundingbox<gpucast::math::point3d>& bbox );
  void                    _parse_material_conf          ( std::istringstream& sstr, gpucast::gl::material& mat) const;
  bool                    _parse_float                  ( std::istringstream& sstr, float& result) const;
  void                    _parse_background             ( std::istringstream& sstr, gpucast::gl::vec3f&) const;

private : // attributes                     
                   
  int                                                                         _argc;
  char**                                                                      _argv;

  bool                                                                        _initialized;
  std::size_t                                                                 _width;
  std::size_t                                                                 _height;

  std::shared_ptr<gpucast::surface_renderer_gl>                               _renderer;
  std::unordered_map<gpucast::surface_renderer::drawable_ptr, std::string>    _objects;

  std::shared_ptr<gpucast::gl::trackball>                                     _trackball;
  gpucast::gl::matrix4f                                                       _projection;
  gpucast::gl::matrix4f                                                       _modelview;
  
  gpucast::math::axis_aligned_boundingbox<gpucast::math::point3d>             _boundingbox;
  gpucast::gl::vec3f                                                          _background;
  bool                                                                        _cullface;
  bool                                                                        _ambient_occlusion;
  bool                                                                        _fxaa;
  
  unsigned                                                                    _frames;
  double                                                                      _time;

  std::shared_ptr<gpucast::gl::program>                                       _fbo_program;
  std::shared_ptr<gpucast::gl::framebufferobject>                             _fbo;
  std::shared_ptr<gpucast::gl::texture2d>                                     _depthattachment;
  std::shared_ptr<gpucast::gl::texture2d>                                     _colorattachment;
  std::shared_ptr<gpucast::gl::sampler>                                       _sample_linear;

  float                                                                       _aoradius;
  unsigned                                                                    _aosamples;
  static const int                                                            TOTAL_RANDOM_SAMPLES = 16384;
  std::shared_ptr<gpucast::gl::texturebuffer>                                 _aosample_offsets;
  std::shared_ptr<gpucast::gl::texture2d>                                     _aorandom_texture;
  std::shared_ptr<gpucast::gl::sampler>                                       _sample_nearest;

  std::shared_ptr<gpucast::gl::plane>                                         _quad;
};



#endif // NURBSVIEW_GLWIDGET_HPP
