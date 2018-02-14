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
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/timer_query.hpp>
#include <gpucast/gl/shaderstoragebuffer.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <gpucast/math/axis_aligned_boundingbox.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_kd.hpp>
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
              contour_map_kd_classification        = 7,
              contour_map_kd_partition             = 8,
              minification                         = 9,
              binary_field                         = 10,
              distance_field                       = 11,  
              prefilter                            = 12,
              count                                = 13 };

  enum aamode {
    disabled = 0,
    prefiltered_edge_estimation = 1,
    prefiltered_curve_estimation = 2,
    pixel_distance_estimation = 3,
    supersampling2x2 = 4,
    supersampling3x3 = 5,
    supersampling4x4 = 6,
    supersampling8x8 = 7 
  };

  struct testrun_t {

    typedef gpucast::beziersurface::trimdomain_ptr domain_ptr;

    struct run_t {
      std::string filename;
      unsigned    index;
      domain_ptr  domain;
      std::size_t size_bytes;
      double      time_ms;
    };

    view               rendermode;
    aamode             antialiasing;
    unsigned           current;
    unsigned           texture_resolution;
    bool               texture_classification;

    std::vector<run_t> data;
    double             preprocessing_ms;    

    void save() {
      std::string filename;

      switch (rendermode) {
      case double_binary_classification: filename = "double_binary_classification"; break;
      case contour_map_binary_classification: filename = "contour_map_binary_classification"; break;
      case contour_map_loop_list_classification: filename = "contour_map_loop_list_classification"; break;
      case original: filename = "original outline"; break;
      case double_binary_partition: filename = "double_binary_partition"; break;
      case contour_map_binary_partition: filename = "contour_map_binary_partition"; break;
      case contour_map_loop_list_partition: filename = "contour_map_loop_list_partition"; break;
      case contour_map_kd_classification: filename = "contour_map_kd_classification"; break;
      case contour_map_kd_partition: filename = "contour_map_kd_partition"; break;
      case minification: filename = "minification"; break;
      case binary_field: filename = "binary_field"; break;
      case distance_field: filename = "distance_field"; break;
      case prefilter: filename = "prefilter"; break;
      default: filename = "unclassified"; break;
      };

      switch (antialiasing) {
      case disabled: filename += "_AA=disabled"; break;
      case prefiltered_edge_estimation: filename += "_AA=prefiltered_edge_estimation"; break;
      case prefiltered_curve_estimation: filename += "_AA=prefiltered_curve_estimation"; break;
      case pixel_distance_estimation: filename += "_AA=pixel_distance_estimation"; break;
      case supersampling2x2: filename += "_AA=supersampling2x2"; break;
      case supersampling3x3: filename += "_AA=supersampling3x3"; break;
      case supersampling4x4: filename += "_AA=supersampling4x4"; break;
      case supersampling8x8: filename += "_AA=supersampling8x8"; break;
      default: filename += "_AA=unclassified"; break;
      };

      filename += "_TexRes=" + std::to_string(texture_resolution);
      filename += "_TexEnabled=" + std::to_string(texture_classification);
      filename += "_UID=" + std::to_string(std::size_t(this));
      
      std::fstream str(filename + ".txt", std::ios::out);

      str << "time preprocessing : " << preprocessing_ms << std::endl;
      str << "# domains : " << data.size() << std::endl;

      auto sort_criteria_curves = [](run_t const& lhs, run_t const& rhs) -> bool { return lhs.domain->curves().size() < rhs.domain->curves().size(); };
      std::sort(data.begin(), data.end(), sort_criteria_curves);

      std::map<unsigned, unsigned> ncurves_count_map;
      std::map<unsigned, unsigned> ncurves_size_map;
      std::map<unsigned, double> ncurves_time_map;
      
      // gather information
      for (auto const& r : data) {
        auto ncurves = r.domain->curves().size();
        ++ncurves_count_map[ncurves];
        ncurves_time_map[ncurves] += r.time_ms;
        ncurves_size_map[ncurves] += r.size_bytes;
      }

      // normalize
      for (auto const& count : ncurves_count_map) {
        ncurves_time_map[count.first] /= count.second;
        str << count.first << " " << ncurves_time_map[count.first] << " " << ncurves_size_map[count.first] << std::endl;
      }

      

    }

    
  };

public : 

  glwidget                ( int argc, char** argv, QGLFormat const& format, QWidget *parent = 0 );
  ~glwidget               ();

  void                    generate_original_view(gpucast::beziersurface::trimdomain_ptr const& domain);
  void                    generate_double_binary_view(gpucast::beziersurface::trimdomain_ptr const& domain);
  void                    generate_bboxmap_view(gpucast::beziersurface::trimdomain_ptr const& domain);
  void                    generate_loop_list_view(gpucast::beziersurface::trimdomain_ptr const& domain);
  void                    generate_kd_view(gpucast::beziersurface::trimdomain_ptr const& domain, gpucast::kd_split_strategy);

  void                    generate_minification_view(gpucast::beziersurface::trimdomain_ptr const& domain, unsigned resolution);
  void                    generate_binary_field(gpucast::beziersurface::trimdomain_ptr const& domain, unsigned resolution);
  void                    generate_distance_field(gpucast::beziersurface::trimdomain_ptr const& domain, unsigned resolution);

  std::size_t             serialize_double_binary(gpucast::beziersurface::trimdomain_ptr const& domain);
  std::size_t             serialize_contour_binary(gpucast::beziersurface::trimdomain_ptr const& domain);
  std::size_t             serialize_contour_kd(gpucast::beziersurface::trimdomain_ptr const& domain, gpucast::kd_split_strategy);
  std::size_t             serialize_contour_loop_list(gpucast::beziersurface::trimdomain_ptr const& domain);

  void                    initialize_sampler();
  void                    generate_trim_region_vbo(gpucast::beziersurface::trimdomain_ptr const& domain);

  void                    add_gl_curve(gpucast::beziersurface::curve_type const& curve, gpucast::math::vec4f const& color);
  void                    add_gl_bbox(gpucast::math::bbox2d const& bbox, gpucast::math::vec4f const& color, bool diagonals = true);

  trimdomain_ptr          get_domain(std::string const& name, std::size_t const index) const;
  std::size_t             get_objects() const;
  std::size_t             get_surfaces(std::string const& name) const;
  void                    pixel_size(unsigned);

public Q_SLOTS : 

  void                    open                          ( std::list<std::string> const& files );
  void                    clear                         ();
  void                    update_view                   ( std::string const& name, std::size_t const index, view current, unsigned resolution );

  void                    show_texel_fetches            (int);
  void                    tex_classification            (int);
  void                    recompile                     ();
  void                    resetview                     ();
  void                    testrun                       (std::list<std::string> const& objects);
  void                    texture_filtering             (int);
  void                    optimal_distance              (int);

  void                    antialiasing                  (enum aamode);
  void                    rendermode                    (enum view);

  void                    zoom                          (float scale);

protected:

  /* virtual */ void      resizeGL                      ( int w, int h );
  /* virtual */ void      paintGL                       ();

  /* virtual */ void      keyPressEvent                 ( QKeyEvent* event);
  /* virtual */ void      keyReleaseEvent               ( QKeyEvent* event);

  /* virtual */ void     	mouseDoubleClickEvent(QMouseEvent * event);
  /* virtual */ void     	mouseMoveEvent(QMouseEvent * event);
  /* virtual */ void     	mousePressEvent(QMouseEvent * event);
  /* virtual */ void     	mouseReleaseEvent(QMouseEvent * event);

private : // helper methods

  void                    _init ();
  void                    _initialize_shader            ();
  void                    _initialize_prefilter         ();

private : // attributes                     
                                            
  int                                               _argc;
  char**                                            _argv;
  bool                                              _initialized;

  int                                               _width;
  int                                               _height;

  std::vector<std::shared_ptr<testrun_t>>           _testruns;

  std::map<std::string, bezierobject_ptr>           _objects;

  std::string                                       _current_object;
  std::size_t                                       _current_surface;

  // simple partition views
  gpucast::math::matrix4f                           _projection;
  std::shared_ptr<gpucast::gl::program>             _partition_program;
  std::vector<std::shared_ptr<gpucast::gl::line>>   _curve_geometry;

  // commonly used resources
  std::unique_ptr<gpucast::gl::texture1d>           _transfertexture;
  unsigned                                          _trimid;
  bool                                              _show_texel_fetches;
  gpucast::math::vec2f                              _domain_size;
  gpucast::math::vec2f                              _domain_min;
  std::unique_ptr<gpucast::gl::plane>               _quad;

  // program to show textures
  std::shared_ptr<gpucast::gl::program>             _tex_program;
  std::shared_ptr<gpucast::gl::program>             _prefilter_program;
  std::unique_ptr<gpucast::gl::texture2d>           _binary_texture;
  std::unique_ptr<gpucast::gl::texture2d>           _distance_field_texture;
  std::unique_ptr<gpucast::gl::texture2d>           _prefilter_texture;

  bool                                              _optimal_distance = false;
  bool                                              _linear_filter = false;
  enum aamode                                       _aamode = disabled;
  unsigned                                          _pixel_size = 1;
  unsigned                                          _current_texresolution = 8;
  bool                                              _tex_classification_enabled = true;

  float                                             _zoom = 1.0f;
  float                                             _shift_x = 0.0f;
  float                                             _shift_y = 0.0f;

  int                                               _last_x = -1;
  int                                               _last_y = -1;
  bool                                              _shift_mode = false;
  bool                                              _zoom_mode = false;

  std::unique_ptr<gpucast::gl::sampler>             _bilinear_filter;
  std::unique_ptr<gpucast::gl::sampler>             _nearest_filter;

  // double binary resources
  std::shared_ptr<gpucast::gl::program>             _db_program;
  std::unique_ptr<gpucast::gl::texturebuffer>       _db_trimdata;
  std::unique_ptr<gpucast::gl::texturebuffer>       _db_celldata;
  std::unique_ptr<gpucast::gl::texturebuffer>       _db_curvelist;
  std::unique_ptr<gpucast::gl::texturebuffer>       _db_curvedata;
  std::unique_ptr<gpucast::gl::texturebuffer>       _db_preclassdata;

  // contourmap_binary resources
  std::shared_ptr<gpucast::gl::program>             _cmb_program;
  std::unique_ptr<gpucast::gl::texturebuffer>       _cmb_partition;
  std::unique_ptr<gpucast::gl::texturebuffer>       _cmb_contourlist;
  std::unique_ptr<gpucast::gl::texturebuffer>       _cmb_curvelist;
  std::unique_ptr<gpucast::gl::texturebuffer>       _cmb_curvedata;
  std::unique_ptr<gpucast::gl::texturebuffer>       _cmb_pointdata;
  std::unique_ptr<gpucast::gl::texturebuffer>       _cmb_preclassdata;

  // contourmap_kd resources
  std::shared_ptr<gpucast::gl::program>             _kd_program;
  std::unique_ptr<gpucast::gl::texturebuffer>       _kd_partition;
  std::unique_ptr<gpucast::gl::texturebuffer>       _kd_contourlist;
  std::unique_ptr<gpucast::gl::texturebuffer>       _kd_curvelist;
  std::unique_ptr<gpucast::gl::texturebuffer>       _kd_curvedata;
  std::unique_ptr<gpucast::gl::texturebuffer>       _kd_pointdata;
  std::unique_ptr<gpucast::gl::texturebuffer>       _kd_preclassdata;

  std::shared_ptr<gpucast::gl::program>             _loop_list_program;
  std::unique_ptr<gpucast::gl::shaderstoragebuffer> _loop_list_loops;
  std::unique_ptr<gpucast::gl::shaderstoragebuffer> _loop_list_contours;
  std::unique_ptr<gpucast::gl::shaderstoragebuffer> _loop_list_curves;
  std::unique_ptr<gpucast::gl::shaderstoragebuffer> _loop_list_points;
  std::unique_ptr<gpucast::gl::texturebuffer>       _loop_list_preclassdata;

  std::unique_ptr<gpucast::gl::timer_query>         _gputimer;

  view                                              _view;
};



#endif // TRIMVIEW_GLWIDGET_HPP
