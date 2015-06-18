/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : mainwindow.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef TRIMVIEW_MAINWINDOW_HPP
#define TRIMVIEW_MAINWINDOW_HPP

#pragma warning(disable: 4127) // Qt conditional expression is constant

#include <glwidget.hpp>

// system includes
#include <GL/glew.h>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QSlider>

#include <slidergroup.hpp>

#include <gpucast/core/nurbssurfaceobject.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>

class mainwindow : public QMainWindow
{
  Q_OBJECT

public: // enums / typdefs

  typedef std::shared_ptr<gpucast::nurbssurfaceobject>   nurbsobject_ptr;
  typedef std::shared_ptr<gpucast::beziersurfaceobject>  bezierobject_ptr;
  typedef std::map<std::string, bezierobject_ptr>        bezierobject_map;

public: // c'tor / d'tor

  mainwindow              ( int argc, char** argv, unsigned width, unsigned height );
  ~mainwindow             ();

  void                    show_drawtime      ( double ms );
  void                    show_memusage      ( std::size_t bytes ) const;
  void                    show_domainsize    ( float umin, float vmin, float umax, float vmax) const;

protected:

  /* virtual */ void      closeEvent   ( QCloseEvent* event );

public Q_SLOTS: // slot events           

  void                    openfile           ();
  void                    update_objectlist  ();
  void                    update_surfacelist ();
  void                    update_view        () const;
  void                    antialiasing       () const;
  void                    pixel_size_changed () const;
  void                    zoom_changed       (int);

private: // methods

private: // attributes

  unsigned                _width;
  unsigned                _height;

  std::map<glwidget::view, std::string> _modes;
  std::map<glwidget::aamode, std::string> _aamodes;
  std::list<double>       _fps;

  // menubar and menubar actions
  QWidget*                _controlwidget;
  QMenu*                  _file_menu;
  QAction*                _action_loadfile;
  QComboBox*              _viewbox;
  QLabel*                 _label_fps;
  QLabel*                 _label_mem;
  QLabel*                 _label_size;
  QPushButton*            _recompile_button;
  QPushButton*            _resetview_button;

  QCheckBox*              _show_texel_fetches;
  QCheckBox*              _linear_texture_filter;
  QCheckBox*              _optimal_distance;
  QComboBox*              _antialiasing;


  QComboBox*              _texture_resolution;
  QComboBox*              _pixel_size;

  SlidersGroup*           _zoom_control;

  bezierobject_map        _objects;
  QListWidget*            _list_object;
  QListWidget*            _list_surface;

  // menu and parameter manipulation
  glwidget*               _glwindow;
  std::list<std::string>  _filenames;
};

#endif // TRIMVIEW_MAINWINDOW_HPP

