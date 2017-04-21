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
#ifndef NURBS_VIEW_MAINWINDOW_HPP
#define NURBS_VIEW_MAINWINDOW_HPP

#pragma warning(disable: 4127) // Qt conditional expression is constant

// system includes
#include <GL/glew.h>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QLabel>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QComboBox>
#include <glwidget.hpp>

#include <gpucast/math/interval.hpp>
#include <slidergroup.hpp>


template <typename map_type>
typename map_type::key_type find_by_second(map_type const& map,
                                             typename map_type::mapped_type const& search_value,
                                             typename map_type::key_type default_value)
{
  auto result = default_value;

  for (auto m : map) {
    if (m.second == search_value) {
      result = m.first;
    }
  }
  return result;
}

class mainwindow : public QMainWindow
{
  Q_OBJECT

public: // c'tor / d'tor

  mainwindow            ( int argc, char** argv, unsigned width, unsigned height );
  ~mainwindow           ();

  void update_interface               ();
  void show_fps                       ( double cputime, double gputime, double postprocess );
  void show_memory_usage              ( gpucast::beziersurfaceobject::memory_usage const& usage );
  void update_count                   ( unsigned triangles, unsigned fragments, unsigned culled_triangles, unsigned trimmed_fragments, unsigned estimate);

protected:

  /* virtual */ void closeEvent       ( QCloseEvent* event );

private slots: // slot events

  void close_window                   ();
  void openfile                       ();
  void addfile                        ();

  void rendering();
  void fillmode();
  void trimming();
  void antialiasing();
  void preclassification();

private: // methods

  void _create_actions      ();
  void _create_widgets      ( int argc, char** argv );
  void _create_menus        ();
  void _create_statusbar    ();

private: // attributes

  unsigned              _width;
  unsigned              _height;

  std::map<gpucast::gl::bezierobject::render_mode, std::string>        _rendering_modes;
  std::map<gpucast::gl::bezierobject::fill_mode, std::string>          _fill_modes;
  std::map<gpucast::beziersurfaceobject::trim_approach_t, std::string> _trimming_modes;
  std::map<gpucast::gl::bezierobject::anti_aliasing_mode, std::string> _antialiasing_modes;
  std::map<unsigned, std::string>                                      _preclassification_modes;

  // menubar and menubar actions
  QMenu*                _file_menu;
  QToolBar*             _file_toolbar;

  QAction*              _action_exit;
  QAction*              _action_loadfile;
  QAction*              _action_addfile;

  QCheckBox*            _checkbox_fxaa;
  QCheckBox*            _checkbox_holefilling;
  QCheckBox*            _checkbox_conservative_rasterization;
  QCheckBox*            _checkbox_vsync;
  QCheckBox*            _checkbox_sao;
  QCheckBox*            _checkbox_diffusemap;
  QCheckBox*            _checkbox_spheremap;
  QCheckBox*            _checkbox_culling;
  QCheckBox*            _checkbox_counting;
  QCheckBox*            _checkbox_tritesselation;

  QLabel*               _counting_result;
  QLabel*               _fps_result;
  QLabel*               _memory_usage;

  SlidersGroup*         _slider_trim_max_bisections;
  FloatSlidersGroup*    _slider_trim_error_tolerance;
  FloatSlidersGroup*    _slider_tesselation_max_pixel_error;
  FloatSlidersGroup*    _slider_tesselation_max_object_error;
  SlidersGroup*         _slider_raycasting_max_iterations;
  FloatSlidersGroup*    _slider_raycasting_error_tolerance;

  QComboBox*            _combobox_rendering;
  QComboBox*            _combobox_fillmode;
  QComboBox*            _combobox_antialiasing;
  QComboBox*            _combobox_trimming;
  QComboBox*            _combobox_preclassification;

  QPushButton*          _button_recompile;
  QPushButton*          _button_set_diffusemap;
  QPushButton*          _button_set_spheremap;

  // menu and parameter manipulation
  glwidget*             _glwindow;

  // menu and parameter manipulation
  QDockWidget*          _menu;

};

#endif // NURBS_VIEW_MAINWINDOW_HPP
