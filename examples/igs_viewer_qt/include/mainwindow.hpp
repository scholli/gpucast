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
#include <QtWidgets/QSlider>
#include <QtWidgets/QLabel>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QComboBox>
#include <glwidget.hpp>

#include <gpucast/math/interval.hpp>


class mainwindow : public QMainWindow
{
  Q_OBJECT

public: // c'tor / d'tor

  mainwindow            ( int argc, char** argv, unsigned width, unsigned height );
  ~mainwindow           ();

  void update_interface               ();
  void show_fps                       ( double cputime, double gputime, double postprocess );

protected:

  /* virtual */ void closeEvent       ( QCloseEvent* event );

private slots: // slot events

  void close_window                   ();
  void openfile                       ();
  void addfile                        ();

  void trimming();
  void antialiasing();

private: // methods

  void _create_actions      ();
  void _create_widgets      ( int argc, char** argv );
  void _create_menus        ();
  void _create_statusbar    ();

private: // attributes

  unsigned              _width;
  unsigned              _height;

  std::map<gpucast::beziersurfaceobject::trim_approach_t, std::string> _trimming_modes;
  std::map<glwidget::antialiasing_mode, std::string> _antialiasing_modes;

  // menubar and menubar actions
  QMenu*                _file_menu;
  QToolBar*             _file_toolbar;

  QAction*              _action_exit;
  QAction*              _action_loadfile;
  QAction*              _action_addfile;

  QCheckBox*            _checkbox_fxaa;
  QCheckBox*            _checkbox_vsync;
  QCheckBox*            _checkbox_sao;
  QCheckBox*            _checkbox_diffusemap;
  QCheckBox*            _checkbox_spheremap;

  QComboBox*            _combobox_antialiasing;
  QComboBox*            _combobox_trimming;

  QPushButton*          _button_recompile;
  QPushButton*          _button_set_diffusemap;
  QPushButton*          _button_set_spheremap;

  // menu and parameter manipulation
  glwidget*             _glwindow;

  // menu and parameter manipulation
  QDockWidget*          _menu;

};

#endif // NURBS_VIEW_MAINWINDOW_HPP
