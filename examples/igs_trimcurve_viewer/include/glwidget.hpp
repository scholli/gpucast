/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : glwidget.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef TRIMVIEW_GLWIDGET_HPP
#define TRIMVIEW_GLWIDGET_HPP

#if WIN32
  #pragma warning(disable: 4127) // Qt conditional expression is constant
  #pragma warning(disable: 4245) // CL warnings
#endif

// system includes
#include <GL/glew.h>

#include <QtOpenGL/QGLWidget>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/primitives/line.hpp>
#include <gpucast/gl/primitives/plane.hpp>
#include <gpucast/gl/texture1d.hpp>
#include <gpucast/gl/shaderstoragebuffer.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <gpucast/math/axis_aligned_boundingbox.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>


class glwidget : public QGLWidget
{
   Q_OBJECT        // must include this if you use Qt signals/slots

public:

  typedef std::shared_ptr<gpucast::beziersurfaceobject> bezierobject_ptr;
  typedef gpucast::beziersurface::trimdomain_ptr        trimdomain_ptr;

  enum view { original                             = 0, 
              double_binary_partition              = 1,
              double_binary_classification         = 2,
              contour_map_binary_partition         = 3, 
              contour_map_binary_classification    = 4, 
              contour_map_loop_list_partition      = 5,
              contour_map_loop_list_classification = 6,
              minification                         = 7,
              count                                = 8 };

public : 

  glwidget                                              ( int argc, char** argv, QGLFormat const& format, QWidget *parent = 0 );
  ~glwidget                                             ();

public Q_SLOTS : 

  void                    open                          ( std::list<std::string> const& files );
  void                    clear                         ();
  void                    update_view                   ( std::string const& name, std::size_t const index, view current );

  void                    generate_original_view        ( gpucast::beziersurface::trimdomain_ptr const& domain );
  void                    generate_double_binary_view   ( gpucast::beziersurface::trimdomain_ptr const& domain );
  void                    generate_bboxmap_view         ( gpucast::beziersurface::trimdomain_ptr const& domain );
  void                    generate_loop_list_view       ( gpucast::beziersurface::trimdomain_ptr const& domain );
  void                    generate_minification_view    ( gpucast::beziersurface::trimdomain_ptr const& domain );

  void                    serialize_double_binary       ( gpucast::beziersurface::trimdomain_ptr const& domain );
  void                    serialize_contour_binary      ( gpucast::beziersurface::trimdomain_ptr const& domain );
  void                    serialize_contour_loop_list   ( gpucast::beziersurface::trimdomain_ptr const& domain );

  void                    add_gl_curve                  ( gpucast::beziersurface::curve_type const& curve, gpucast::math::vec4f const& color  );
  void                    add_gl_bbox                   ( gpucast::math::bbox2d const& bbox, gpucast::math::vec4f const& color  );

  trimdomain_ptr          get_domain                    ( std::string const& name, std::size_t const index ) const;
  std::size_t             get_objects                   () const;
  std::size_t             get_surfaces                  ( std::string const& name ) const;

  void                    show_texel_fetches            (bool);
  void                    recompile                     ();

protected:

  /* virtual */ void      initializeGL                  ();
  /* virtual */ void      resizeGL                      ( int w, int h );
  /* virtual */ void      paintGL                       ();

  /* virtual */ void      keyPressEvent                 ( QKeyEvent* event);
  /* virtual */ void      keyReleaseEvent               ( QKeyEvent* event);

private : // helper methods

  void                    _init ();
  void                    _initialize_shader            ();

private : // attributes                     
                                            
  int                                     _argc;
  char**                                  _argv;
  bool                                    _initialized;

  int                                     _width;
  int                                     _height;

  std::map<std::string, bezierobject_ptr> _objects;

  std::string                             _current_object;
  std::size_t                             _current_surface;

  // simple partition views
  gpucast::math::matrix4f                        _projection;
  std::shared_ptr<gpucast::gl::program>          _partition_program;
  std::vector<gpucast::gl::line*>                _curve_geometry;

  // commonly used resources
  gpucast::gl::texture1d*                        _transfertexture;
  unsigned                                       _trimid;
  bool                                           _show_texel_fetches;
  gpucast::math::vec2f                           _domain_size;
  gpucast::math::vec2f                           _domain_min;
  gpucast::gl::plane*                            _quad;

  // double binary resources
  std::shared_ptr<gpucast::gl::program>          _db_program;
  gpucast::gl::texturebuffer*                    _db_trimdata;
  gpucast::gl::texturebuffer*                    _db_celldata;
  gpucast::gl::texturebuffer*                    _db_curvelist;
  gpucast::gl::texturebuffer*                    _db_curvedata;

  // contourmap_binary resources
  std::shared_ptr<gpucast::gl::program>          _cmb_program;
  gpucast::gl::texturebuffer*                    _cmb_partition;
  gpucast::gl::texturebuffer*                    _cmb_contourlist;
  gpucast::gl::texturebuffer*                    _cmb_curvelist;
  gpucast::gl::texturebuffer*                    _cmb_curvedata;
  gpucast::gl::texturebuffer*                    _cmb_pointdata;

  // contourmap_kd resources
  std::shared_ptr<gpucast::gl::program>          _kd_program;
  gpucast::gl::texturebuffer*                    _kd_partition;
  gpucast::gl::texturebuffer*                    _kd_contourlist;
  gpucast::gl::texturebuffer*                    _kd_curvelist;
  gpucast::gl::texturebuffer*                    _kd_curvedata;
  gpucast::gl::texturebuffer*                    _kd_pointdata;

  std::shared_ptr<gpucast::gl::program>          _loop_list_program;
  gpucast::gl::shaderstoragebuffer*              _loop_list_loops;
  gpucast::gl::shaderstoragebuffer*              _loop_list_contours;
  gpucast::gl::shaderstoragebuffer*              _loop_list_curves;
  gpucast::gl::shaderstoragebuffer*              _loop_list_points;

  view                                           _view;
};



#endif // TRIMVIEW_GLWIDGET_HPP
