/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : mainwindow.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "mainwindow.hpp"

#pragma warning(disable: 4127) // Qt conditional expression is constant

// system includes
#include <QtGui/QCloseEvent>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QStatusBar>  
#include <QtGui/QMenuBar>
#include <QtGui/QAction>
#include <QtCore/QString>

#include <iostream>

#include <gpucast/gl/error.hpp>


///////////////////////////////////////////////////////////////////////////////
mainwindow::mainwindow(int argc, char** argv, unsigned width, unsigned height)
  : _width ( width ),
    _height ( height )
{
  _create_widgets( argc, argv );
  _create_menus();
  _create_statusbar();
  _create_actions();

  setUnifiedTitleAndToolBarOnMac(true);

  update_interface();
}


///////////////////////////////////////////////////////////////////////////////
mainwindow::~mainwindow()
{}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::closeEvent(QCloseEvent* /*event*/)
{}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::close_window()
{
  hide();
  close();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::update_interface ( )
{
  // update window
  QMainWindow::update();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::show_fps ( double fps )
{
  QString message = "fps : ";
  message.append(QString("%1").arg(_glwindow->width()));
  message.append("x");
  message.append(QString("%1").arg(_glwindow->height()));
  message.append(" @ ");
  message.append(QString("%1").arg(fps));
  statusBar()->showMessage(message);
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::openfile()
{
 QStringList qfilelist = QFileDialog::getOpenFileNames(this, tr("Open Volume"), ".", tr("Surface Files (*.igs *.cfg *.dat)"));
 std::list<std::string> filelist;
 std::transform(qfilelist.begin(), qfilelist.end(), std::back_inserter(filelist), []( QString const& qstr ) { return qstr.toStdString(); } );
 _glwindow->open(filelist); 
 update_interface();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::addfile()
{
 QStringList qfilelist = QFileDialog::getOpenFileNames(this, tr("Open Volume"), ".", tr("Surface Files (*.igs *.cfg)"));
 std::list<std::string> filelist;
 std::transform(qfilelist.begin(), qfilelist.end(), std::back_inserter(filelist), []( QString const& qstr ) { return qstr.toStdString(); } );
 _glwindow->add(filelist); 
 update_interface();
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_actions()
{
  _action_exit                            = new QAction(tr("Exit"), this);
  _action_loadfile                        = new QAction(tr("Open"), this);
  _action_addfile                         = new QAction(tr("Add"),  this);

  // menu
  connect(_action_exit,                     SIGNAL( triggered()), this,         SLOT( close_window() ));
  connect(_action_loadfile,                 SIGNAL( triggered()), this,         SLOT( openfile() ));
  connect(_action_addfile,                  SIGNAL( triggered()), this,         SLOT( addfile() ));

  connect(_button_recompile,                SIGNAL( released() ), _glwindow,    SLOT( recompile() ));
  connect(_button_set_diffusemap,           SIGNAL( released() ), _glwindow,    SLOT( load_diffusemap() ));
  connect(_button_set_spheremap,            SIGNAL( released() ), _glwindow,    SLOT( load_spheremap() ));

  connect(_checkbox_spheremap,              SIGNAL( stateChanged(int) ), _glwindow,    SLOT( spheremapping(int) ));
  connect(_checkbox_diffusemap,             SIGNAL( stateChanged(int) ), _glwindow,    SLOT( diffusemapping(int) ));
  connect(_checkbox_fxaa,                   SIGNAL( stateChanged(int) ), _glwindow,    SLOT( fxaa(int) ));
  connect(_checkbox_vsync,                  SIGNAL( stateChanged(int) ), _glwindow,    SLOT( vsync(int) ));
  connect(_checkbox_sao,                    SIGNAL( stateChanged(int) ), _glwindow,    SLOT( ambient_occlusion(int) ));

  _file_menu->addSeparator();
  _file_menu->addAction   (_action_loadfile);
  _file_menu->addAction   (_action_addfile);
  _file_menu->addAction   (_action_exit);
}
      

///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_widgets( int argc, char** argv )
{
  /////////////////////////////////////
  // create QGLWidget as central element
  /////////////////////////////////////
  QGLFormat context_options;

  context_options.setVersion ( 4, 2 );
  //context_options.setProfile ( QGLFormat::CompatibilityProfile );
  context_options.setProfile ( QGLFormat::CoreProfile );
  context_options.setRgba    ( true );

  _glwindow = new glwidget   ( argc, argv, context_options, this );

  resize(_width+124, _height+41);
  setCentralWidget(_glwindow);
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_menus()
{
  _file_menu = menuBar()->addMenu(tr("File"));
  
  _menu = new QDockWidget(this);
  _menu->setFixedWidth(120);
  this->addDockWidget(Qt::RightDockWidgetArea, _menu);

  _button_recompile  = new QPushButton("Compile Shaders", _menu);
  _button_recompile->move  (5, 30);

  _button_set_spheremap  = new QPushButton("Spheremap", _menu);
  _button_set_spheremap->move  (5, 60);

  _checkbox_spheremap    = new QCheckBox("Enable Spheremapping", _menu);
  _checkbox_spheremap->move    (5, 90);

  _button_set_diffusemap = new QPushButton("Diffusemap", _menu);
  _button_set_diffusemap->move (5, 120);
  
  _checkbox_diffusemap   = new QCheckBox("Enable Diffusemapping", _menu);
  _checkbox_diffusemap->move   (5, 150);

  _checkbox_fxaa   = new QCheckBox("Enable FXAA", _menu);
  _checkbox_fxaa->move   (5, 180);

  _checkbox_vsync   = new QCheckBox("Enable VSync", _menu);
  _checkbox_vsync->move   (5, 210);

  _checkbox_sao   = new QCheckBox("Enable Ambient Occlusion", _menu);
  _checkbox_sao->move   (5, 240);
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_statusbar()
{}




