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

#include <list>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <glpp/vertexarrayobject.hpp>
#include <glpp/elementarraybuffer.hpp>
#include <glpp/arraybuffer.hpp>
#include <glpp/program.hpp>
#include <glpp/sampler.hpp>
#include <glpp/texturebuffer.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/framebufferobject.hpp>
#include <glpp/renderbuffer.hpp>
#include <glpp/primitives/plane.hpp>

#include <glpp/math/matrix4x4.hpp>
#include <glpp/math/vec3.hpp>

#include <tml/axis_aligned_boundingbox.hpp>
#include <gpucast/beziersurfaceobject.hpp>
#include <gpucast/surface_renderer_gl.hpp>



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

  void                    _openfile                     ( std::string const& file, tml::axis_aligned_boundingbox<tml::point3d>& bbox );
  void                    _parse_material_conf          ( std::istringstream& sstr, glpp::material& mat) const;
  bool                    _parse_float                  ( std::istringstream& sstr, float& result) const;
  void                    _parse_background             ( std::istringstream& sstr, glpp::vec3f&) const;

private : // attributes                     
                   
  int                                                 _argc;
  char**                                              _argv;

  bool                                                _initialized;
  std::size_t                                         _width;
  std::size_t                                         _height;

  boost::shared_ptr<gpucast::surface_renderer_gl>     _renderer;
  boost::unordered_map<gpucast::surface_renderer::drawable_ptr, std::string>  _objects;

  boost::shared_ptr<glpp::trackball>                  _trackball;
  glpp::matrix4f                                      _projection;
  glpp::matrix4f                                      _modelview;
  
  tml::axis_aligned_boundingbox<tml::point3d>         _boundingbox;
  glpp::vec3f                                         _background;
  bool                                                _cullface;
  bool                                                _ambient_occlusion;
  bool                                                _fxaa;
  
  unsigned                                            _frames;
  double                                              _time;

  boost::shared_ptr<glpp::program>                    _fbo_program;
  boost::shared_ptr<glpp::framebufferobject>          _fbo;
  boost::shared_ptr<glpp::texture2d>                  _depthattachment;
  boost::shared_ptr<glpp::texture2d>                  _colorattachment;
  boost::shared_ptr<glpp::sampler>                    _sample_linear;

  float                                               _aoradius;
  unsigned                                            _aosamples;
  static const int                                    TOTAL_RANDOM_SAMPLES = 16384;
  boost::shared_ptr<glpp::texturebuffer>              _aosample_offsets;
  boost::shared_ptr<glpp::texture2d>                  _aorandom_texture;
  boost::shared_ptr<glpp::sampler>                    _sample_nearest;

  boost::shared_ptr<glpp::plane>                      _quad;
};



#endif // NURBSVIEW_GLWIDGET_HPP
