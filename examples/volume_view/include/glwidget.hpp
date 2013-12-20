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
#ifndef VOLUME_VIEW_GLWIDGET_HPP
#define VOLUME_VIEW_GLWIDGET_HPP

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

#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>
#include <boost/scoped_ptr.hpp>

#include <glpp/vertexarrayobject.hpp>
#include <glpp/elementarraybuffer.hpp>
#include <glpp/arraybuffer.hpp>
#include <glpp/program.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/primitives/coordinate_system.hpp>

#include <glpp/util/transformation_sequence.hpp>

#include <glpp/math/matrix4x4.hpp>

#include <tml/axis_aligned_boundingbox.hpp>
#include <gpucast/nurbsvolumeobject.hpp>
#include <gpucast/beziervolumeobject.hpp>
#include <gpucast/surface_renderer_gl.hpp>
#include <gpucast/isosurface/splat/isosurface_renderer_splatbased.hpp>
#include <gpucast/isosurface/octree/isosurface_renderer_octreebased.hpp>
#include <gpucast/isosurface/fragment/isosurface_renderer_unified_sampling.hpp>
#include <gpucast/isosurface/fragment/isosurface_renderer_interval_sampling.hpp>
#include <gpucast/volume_renderer_tesselator.hpp>



class glwidget : public QGLWidget
{
   Q_OBJECT        // must include this if you use Qt signals/slots

public:

  enum rendermode_t {
                    splatting,
                    octree_isosurface,
                    grid_isosurface,
                    tesselator,
                    face_interval_raycasting,
                    unified_sampling
                  };

  struct render_settings 
  {
    bool        fxaa;
    bool        cullface;
    bool        isosides;

    bool        sample_based_face_intersection;
    bool        detect_implicit_inflection;
    bool        detect_implicit_extremum;

    unsigned    newton_iterations;
    float       newton_epsilon;
    bool        newton_screenspace_epsilon;

    unsigned    octree_max_depth;
    unsigned    octree_max_volumes_per_node;

    float       nearplane;
    float       farplane;

    unsigned    isosearch_max_binary_steps;
    bool        isosearch_adaptive_sampling;
    float       isosearch_relative_value;
    float       isosearch_min_sample_distance;
    float       isosearch_max_sample_distance;
    float       isosearch_sample_scale;

    float       transparency_surface;
    float       transparency_isosurface;

    gpucast::visualization_properties visualization_props;
  };

  typedef boost::shared_ptr<gpucast::volume_renderer>           isosurface_renderer_ptr;
  typedef boost::shared_ptr<gpucast::surface_renderer_gl>       surface_renderer_ptr;

  typedef std::map<rendermode_t, isosurface_renderer_ptr>       modemap;
  typedef modemap::value_type                                   modepair;
  typedef gpucast::beziervolumeobject::boundingbox_type         boundingbox_t;
  
public : 

  glwidget                                            ( int argc, char** argv, QGLFormat const& format, QWidget *parent = 0 );
  ~glwidget                                           ();

public Q_SLOTS : 

  void                    recompile                     ();

  isosurface_renderer_ptr const& isosurface_renderer    () const;
  surface_renderer_ptr    const& surface_renderer       () const;

  void                    reset_trackball               ();
  void                    boundingbox                   ( boundingbox_t const& b );

  void                    apply                         ( render_settings const&);

  void                    apply                         ( render_settings const&,
                                                          boost::shared_ptr<gpucast::nurbsvolumeobject> const& nurbsobject,
                                                          boost::shared_ptr<gpucast::beziervolumeobject> const& bezierobject,
                                                          std::string const& attribute,
                                                          std::string const& filename,
                                                          rendermode_t mode );

  void                    start_record                  ();
  void                    stop_record                   ( std::string const& file );
  void                    run_sequence                  ( std::string const& sequence_file, std::string const& result_file );
  void                    abort_sequence                ();

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
  void                    _init_shader                  ();
  void                    _create_shader                ();
  void                    _print_contextinfo            ();

  void                    _change_isosurface_renderer   ( rendermode_t mode );

private : // attributes                     
                                            
  int                                                 _argc;
  char**                                              _argv;

  bool                                                _initialized;
  bool                                                _fxaa;

  int                                                 _width;
  int                                                 _height;

  isosurface_renderer_ptr                             _isosurface_renderer;
  surface_renderer_ptr                                _surface_renderer;

  boost::shared_ptr<glpp::trackball>                  _trackball;
  glpp::matrix4f                                      _projection;
  glpp::matrix4f                                      _modelview;
  
  boundingbox_t                                       _boundingbox;

  unsigned                                            _frames;
  double                                              _time;
  
  boost::shared_ptr<glpp::coordinate_system>          _coordinate_system;

  boost::shared_ptr<glpp::program>                    _base_program;
  boost::shared_ptr<glpp::program>                    _fbo_program;
  boost::shared_ptr<glpp::program>                    _depth_copy_program;

  boost::shared_ptr<glpp::framebufferobject>          _fxaa_input_fbo;
  boost::shared_ptr<glpp::texture2d>                  _fxaa_input_color;  
  boost::shared_ptr<glpp::texture2d>                  _fxaa_input_depth;

  boost::shared_ptr<glpp::framebufferobject>          _color_depth_fbo;
  boost::shared_ptr<glpp::texture2d>                  _color_depth_texture;

  boost::shared_ptr<glpp::framebufferobject>          _surface_fbo;
  boost::shared_ptr<glpp::texture2d>                  _surface_color;  
  boost::shared_ptr<glpp::texture2d>                  _surface_depth;

  boost::shared_ptr<glpp::plane>                      _quad;

  // performance testing
  glpp::transformation_sequence                       _test_sequence;
  std::vector<double>                                 _test_sequence_drawtimes;
  bool                                                _run_sequence;
  bool                                                _record_sequence;
  std::string                                         _test_sequence_filename;
  std::string                                         _test_results_filename;
};



#endif // VOLUME_VIEW_GLWIDGET_HPP
