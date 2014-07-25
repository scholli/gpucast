/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : mainwindow.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef VOLUME_VIEW_MAINWINDOW_HPP
#define VOLUME_VIEW_MAINWINDOW_HPP

#pragma warning(disable: 4127) // Qt conditional expression is constant

// system includes
#include <GL/glew.h>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QSlider>
#include <QtWidgets/QLabel>
#include <QtWidgets/QTextedit>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QLineEdit>

#include <glwidget.hpp>

#include <gpucast/math/interval.hpp>

#include <gpucast/gl/util/timer.hpp>

#include <gpucast/volume/volume_converter.hpp>




class mainwindow : public QMainWindow
{
  Q_OBJECT

public: // enums / typdefs

  typedef std::shared_ptr<gpucast::nurbsvolumeobject>  nurbsvolumeobject_ptr;
  typedef std::shared_ptr<gpucast::beziervolumeobject> beziervolumeobject_ptr;

public: // c'tor / d'tor

  mainwindow            ( int argc, char** argv, unsigned width, unsigned height );
  ~mainwindow           ();

  void update_interface                 ();
  double frame                          ();
  void show_fps                         ( double fps );
  void update_attributelist             ();

protected:

  /* virtual */ void closeEvent         ( QCloseEvent* event );
                                        
private slots: // slot events           
           
  void apply_default_settings             ();
  void apply_settings_to_interface        ();
  void apply_settings_to_renderer         ();
  void apply_volume_to_renderer           ();
  void apply_interface_to_renderer        ();

  void apply_attributes_to_interface      ();

  void close_window                       ();
  void open_volume_file                   ();
  void open_surface_file                  ();
                                      
  void change_adaptive_sampling           ();
  void change_show_isosides               ();
  void change_fxaa                        ();
  void change_vsync                       ();
  void change_backface_culling            ();

  void change_show_face_samples           ();
  void change_show_face_intersections     ();
  void change_show_face_intersection_tests();
  void change_newton_inflection           ();
  void change_newton_extremum             ();
  void change_sample_based_face_intersection ();
                                          
  void change_min_sample_distance         ();
  void change_max_sample_distance         ();
  void change_adaptive_sample_scale       ();
                                          
  void change_max_binary_searches         ();
  void change_epsilon_newton_iteration    ();
  void change_max_newton_iteration        ();
  void change_relative_isovalue           ();
  void change_screenspace_newton_error    ();
  void change_isosurface_transparency     ();
  void change_surface_transparency        ();

  void change_rendermode                  ( QString const& mode );
  void change_current_attribute           ( QString const& attribute );

  void stop_record                        ();
  void run_sequence                       ();
 
private: // methods

  glwidget::rendermode_t  _get_mode     ( QString const& mode ) const;

  void _create_actions                  ();
  void _create_widgets                  ( int argc, char** argv );
  void _create_menus                    ();
  void _create_statusbar                ();

private: // attributes

  glwidget::render_settings                     _settings;
  gpucast::volume_converter                     _volume_converter;

  unsigned              _width;
  unsigned              _height;

  // menubar and menubar actions
  QMenu*                _file_menu;
  QToolBar*             _file_toolbar;
  QAction*              _action_exit;
  QAction*              _action_loadfile;
  QAction*              _action_loadsurffile;
  QAction*              _action_start_testing;

  // menu and parameter manipulation
  QDockWidget*          _menu;

  QCheckBox*            _checkbox_adaptive_sampling;
  QCheckBox*            _checkbox_screenspace_newton_error;
  QCheckBox*            _checkbox_backface_culling;
  QCheckBox*            _checkbox_show_isosides;
  QCheckBox*            _checkbox_fxaa;
  QCheckBox*            _checkbox_vsync;

  QCheckBox*            _checkbox_show_face_samples;
  QCheckBox*            _checkbox_show_face_intersections;
  QCheckBox*            _checkbox_show_face_intersection_tests;
  QCheckBox*            _checkbox_newton_inflection;
  QCheckBox*            _checkbox_newton_extremum;
  QCheckBox*            _checkbox_sample_based_face_intersection;

  QSlider*              _slider_min_sample_distance;
  QSlider*              _slider_max_sample_distance;
  QSlider*              _slider_adaptive_sample_scale;

  QSlider*              _slider_max_binary_searches;
  QSlider*              _slider_epsilon_newton_iteration;
  QSlider*              _slider_max_newton_iteration;
  QSlider*              _slider_relative_isovalue;
  QSlider*              _slider_isosurface_transparency;
  QSlider*              _slider_surface_transparency;

  QLabel*               _label_fps;
  QLabel*               _label_adaptive_sampling;
  QLabel*               _label_screenspace_newton_error;
  QLabel*               _label_backface_culling;
  QLabel*               _label_show_isosides;
  QLabel*               _label_fxaa;
  QLabel*               _label_vsync;
  QLabel*               _label_min_sample_distance;
  QLabel*               _label_max_sample_distance;
  QLabel*               _label_adaptive_sample_scale;
  QLabel*               _label_max_binary_searches;
  QLabel*               _label_epsilon_newton_iteration;
  QLabel*               _label_max_newton_iteration;
  QLabel*               _label_relative_isovalue;
  QLabel*               _label_isosurface_transparency;
  QLabel*               _label_surface_transparency;

  QLabel*               _label_show_face_samples;
  QLabel*               _label_show_face_intersections;
  QLabel*               _label_show_face_intersection_tests;
  QLabel*               _label_newton_inflection;
  QLabel*               _label_newton_extremum;
  QLabel*               _label_sample_based_face_intersection;

  QLabel*               _edit_min_sample_distance;
  QLabel*               _edit_max_sample_distance;
  QLabel*               _edit_adaptive_sample_scale;
  QLabel*               _edit_max_binary_searches;
  QLabel*               _edit_epsilon_newton_iteration;
  QLabel*               _edit_max_newton_iteration;
  QLabel*               _edit_relative_isovalue;
  QLabel*               _edit_isosurface_transparency;
  QLabel*               _edit_surface_transparency;

  QComboBox*            _box_choose_rendermode;
  QComboBox*            _box_choose_attribute;

  QPushButton*          _button_start_sequence;
  QPushButton*          _button_stop_sequence;
  QPushButton*          _button_run_sequence;
  QPushButton*          _button_abort_sequence;
  QPushButton*          _button_recompile;
  QPushButton*          _button_apply;

  QLineEdit*            _lineedit_sequence_file;
  QLineEdit*            _lineedit_test_file;

  // gl widget
  gpucast::gl::timer                                       _timer;
  unsigned                                          _frames;
  glwidget*                                         _glwindow;

  nurbsvolumeobject_ptr                             _nurbsobject;
  beziervolumeobject_ptr                            _bezierobject;
  std::string                                       _current_file_name;

  std::map<glwidget::rendermode_t, QString>         _modemap;

  unsigned                                          _slider_width;
  gpucast::math::interval<float>                              _interval_min_sample_distance;
  gpucast::math::interval<float>                              _interval_max_sample_distance;
  gpucast::math::interval<float>                              _interval_adaptive_sample_scale;
  gpucast::math::interval<unsigned>                           _interval_max_binary_searches;
  gpucast::math::interval<float>                              _interval_epsilon_newton_iteration;
  gpucast::math::interval<unsigned>                           _interval_max_newton_iteration;
  gpucast::math::interval<float>                              _interval_relative_isovalue;
  gpucast::math::interval<float>                              _interval_transparency;
};

#endif // VOLUME_VIEW_MAINWINDOW_HPP

