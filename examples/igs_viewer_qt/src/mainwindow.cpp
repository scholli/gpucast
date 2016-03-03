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
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QStatusBar>  
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QAction>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QBoxLayout>
#include <QtCore/QString>

#include <iostream>

#include <gpucast/gl/error.hpp>


///////////////////////////////////////////////////////////////////////////////
mainwindow::mainwindow(int argc, char** argv, unsigned width, unsigned height)
  : _width ( width ),
    _height ( height )
{
  /////////////////////////////////////
  // anti-aliasing modes
  /////////////////////////////////////
  _antialiasing_modes.insert(std::make_pair(glwidget::disabled, "No Anti-Aliasing"));
  _antialiasing_modes.insert(std::make_pair(glwidget::prefiltered_edge_estimation, "Prefiltered Edge Estimation"));
  _antialiasing_modes.insert(std::make_pair(glwidget::supersampling2x2, "Supersampling(2x2)"));
  _antialiasing_modes.insert(std::make_pair(glwidget::supersampling3x3, "Supersampling(3x3)"));
  _antialiasing_modes.insert(std::make_pair(glwidget::supersampling4x4, "Supersampling(4x4)"));
  _antialiasing_modes.insert(std::make_pair(glwidget::supersampling8x8, "Supersampling(8x8)"));

  /////////////////////////////////////
  // trimming modes
  /////////////////////////////////////
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::no_trimming, "No Trimming"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::curve_binary_partition, "Classic double binary partition"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::contour_binary_partition, "Contour binary-partition"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::contour_kd_partition, "Contour kd-partition"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::contour_list, "Contour loop-list"));

  /////////////////////////////////////
  // create GUI
  /////////////////////////////////////
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
  trimming();
  antialiasing();

  // update window
  QMainWindow::update();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::show_fps ( double cputime, double gputime, double postprocess )
{
  QString message = "fps : ";
  message.append(QString("%1").arg(_glwindow->width()));
  message.append("x");
  message.append(QString("%1").arg(_glwindow->height()));
  message.append(QString(" ||| CPU=%1 ms ").arg(cputime));
  message.append(QString(" ||| Draw=%1 ms ").arg(gputime));
  message.append(QString(" ||| Postprocess=%1 ms ").arg(postprocess));
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
void mainwindow::trimming()
{
  std::string str = _combobox_trimming->currentText().toStdString();
  gpucast::beziersurfaceobject::trim_approach_t trimapproach = gpucast::beziersurfaceobject::no_trimming;

  for (auto mode : _trimming_modes)
  {
    if (mode.second == str) 
      trimapproach = mode.first;
  }

  _glwindow->trimming(trimapproach);
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::antialiasing()
{
  std::string str = _combobox_trimming->currentText().toStdString();
  glwidget::antialiasing_mode aa_mode = glwidget::disabled;

  for (auto mode : _antialiasing_modes)
  {
    if (mode.second == str)
      aa_mode = mode.first;
  }

  _glwindow->antialiasing(aa_mode);
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

  connect(_combobox_antialiasing, SIGNAL(currentIndexChanged(int)), this, SLOT(antialiasing()));
  connect(_combobox_trimming, SIGNAL(currentIndexChanged(int)), this, SLOT(trimming()));

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

  context_options.setVersion ( 4, 4 );
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

  auto widget = new QWidget{ _menu };
  _menu->setWidget(widget);
  _menu->setFixedWidth(600);
  _menu->setFixedHeight(1200);
  this->addDockWidget(Qt::RightDockWidgetArea, _menu);

  auto layout = new QGridLayout;
  layout->setAlignment(Qt::AlignTop);
  widget->setLayout(layout);

  _button_recompile      = new QPushButton("Recompile Shaders", _menu);
  _button_set_spheremap  = new QPushButton("Choose Spheremap", _menu);
  _button_set_diffusemap = new QPushButton("Choose Diffusemap", _menu);
  _checkbox_spheremap    = new QCheckBox("Enable Spheremapping", _menu);
  _checkbox_diffusemap   = new QCheckBox("Enable Diffusemapping", _menu);
  _checkbox_fxaa         = new QCheckBox("Enable FXAA", _menu);
  _checkbox_vsync        = new QCheckBox("Enable VSync", _menu);
  _checkbox_sao          = new QCheckBox("Enable Ambient Occlusion", _menu);
  
  _combobox_antialiasing = new QComboBox;
  _combobox_trimming     = new QComboBox;

  std::for_each(_antialiasing_modes.begin(), _antialiasing_modes.end(), [&](std::map<glwidget::antialiasing_mode, std::string>::value_type const& p) { _combobox_antialiasing->addItem(p.second.c_str()); });
  std::for_each(_trimming_modes.begin(), _trimming_modes.end(), [&](std::map<gpucast::beziersurfaceobject::trim_approach_t, std::string>::value_type const& p) { _combobox_trimming->addItem(p.second.c_str()); });

  // apply widget into layout
  unsigned row = 0;
  layout->addWidget(new QLabel("==========System=========="), row++, 0);
  layout->addWidget(_button_recompile, row++, 0);
  layout->addWidget(_checkbox_vsync, row++, 0);

  layout->addWidget(new QWidget, row++, 0);
  layout->addWidget(new QLabel("=======Anti-Aliasing======="), row++, 0);
  layout->addWidget(_checkbox_fxaa, row++, 0);
  layout->addWidget(_combobox_antialiasing, row++, 0);

  layout->addWidget(new QLabel("=======Trimming======="), row++, 0);
  layout->addWidget(_combobox_trimming, row++, 0);

  layout->addWidget(new QLabel("=========Rendering========="), row++, 0);

  layout->addWidget(new QLabel("==========Shading=========="), row++, 0);
  layout->addWidget(_checkbox_spheremap, row++, 0);
  layout->addWidget(_button_set_spheremap, row++, 0);
  layout->addWidget(_checkbox_diffusemap, row++, 0);
  layout->addWidget(_button_set_diffusemap, row++, 0);
  layout->addWidget(_checkbox_sao, row++, 0);
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_statusbar()
{}




