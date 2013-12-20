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
#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>
#include <QtGui/QListWidget>
#include <QtGui/QDockWidget>
#include <QtGui/QComboBox>
#include <QtGui/QLabel>

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

protected:

  /* virtual */ void      closeEvent   ( QCloseEvent* event );

private Q_SLOTS: // slot events           

  void                    openfile           ();
  void                    update_objectlist  ();
  void                    update_surfacelist ();
  void                    update_view        () const;

private: // methods

private: // attributes

  unsigned                _width;
  unsigned                _height;

  std::map<glwidget::view, std::string> _modes;
  std::list<double>       _fps;

  // menubar and menubar actions
  QWidget*                _controlwidget;
  QMenu*                  _file_menu;
  QAction*                _action_loadfile;
  QComboBox*              _viewbox;
  QLabel*                 _label_fps;
  QLabel*                 _label_mem;

  bezierobject_map        _objects;
  QListWidget*            _list_object;
  QListWidget*            _list_surface;

  // menu and parameter manipulation
  glwidget*               _glwindow;
  std::list<std::string>  _filenames;
};

#endif // TRIMVIEW_MAINWINDOW_HPP

