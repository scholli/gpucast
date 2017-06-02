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
  // surface rendering modes
  /////////////////////////////////////
  _rendering_modes.insert(std::make_pair(gpucast::gl::bezierobject::raycasting, "Ray Casting"));
  _rendering_modes.insert(std::make_pair(gpucast::gl::bezierobject::tesselation, "Tesselation"));
  _rendering_modes.insert(std::make_pair(gpucast::gl::bezierobject::shadow, "ShadowMode"));
  _rendering_modes.insert(std::make_pair(gpucast::gl::bezierobject::shadow_lowres, "ShadowMode LowQuality"));
  
  /////////////////////////////////////
  // fill modes
  /////////////////////////////////////
  _fill_modes.insert(std::make_pair(gpucast::gl::bezierobject::solid, "Solid"));
  _fill_modes.insert(std::make_pair(gpucast::gl::bezierobject::wireframe, "Wireframe"));
  _fill_modes.insert(std::make_pair(gpucast::gl::bezierobject::points, "Points"));

  /////////////////////////////////////
  // anti-aliasing modes
  /////////////////////////////////////
  _antialiasing_modes.insert(std::make_pair(gpucast::gl::bezierobject::disabled, "No Anti-Aliasing"));
  _antialiasing_modes.insert(std::make_pair(gpucast::gl::bezierobject::prefiltered_edge_estimation, "Prefiltered Edge Estimation"));
  _antialiasing_modes.insert(std::make_pair(gpucast::gl::bezierobject::multisampling2x2, "Multisampling(2x2)"));
  _antialiasing_modes.insert(std::make_pair(gpucast::gl::bezierobject::multisampling3x3, "Multisampling(3x3)"));
  _antialiasing_modes.insert(std::make_pair(gpucast::gl::bezierobject::multisampling4x4, "Multisampling(4x4)"));
  _antialiasing_modes.insert(std::make_pair(gpucast::gl::bezierobject::multisampling8x8, "Multisampling(8x8)"));
  _antialiasing_modes.insert(std::make_pair(gpucast::gl::bezierobject::msaa, "MSAA"));

  /////////////////////////////////////
  // trimming modes
  /////////////////////////////////////
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::no_trimming, "No Trimming"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::curve_binary_partition, "Classic double binary partition"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::contour_binary_partition, "Contour binary-partition"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::contour_kd_partition, "Contour kd-partition"));
  _trimming_modes.insert(std::make_pair(gpucast::beziersurfaceobject::contour_list, "Contour loop-list"));

  /////////////////////////////////////
  // preclassification modes
  /////////////////////////////////////
  _preclassification_modes.insert(std::make_pair(0, "No Preclassification"));
  _preclassification_modes.insert(std::make_pair(4, "Preclassification 4x4"));
  _preclassification_modes.insert(std::make_pair(8, "Preclassification 8x8"));
  _preclassification_modes.insert(std::make_pair(16, "Preclassification 16x16"));
  _preclassification_modes.insert(std::make_pair(32, "Preclassification 32x32"));
  _preclassification_modes.insert(std::make_pair(64, "Preclassification 64x64"));
  _preclassification_modes.insert(std::make_pair(128, "Preclassification 128x128"));

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

  // update window
  QMainWindow::update();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::show_fps ( double cputime, double gputime, double postprocess )
{
  QString message = "IGS Viewer || fps=";
  message.append(QString("%1").arg(1000.0f/gputime));
  message.append(tr(" Hz"));
  message.append(QString(" || Size : %1").arg(_glwindow->width()));
  message.append("x");
  message.append(QString("%1").arg(_glwindow->height()));
  message.append(QString(" || CPU=%1 ms ").arg(cputime));
  message.append(QString(" || Draw=%1 ms ").arg(gputime));
  message.append(QString(" || PostDraw=%1 ms ").arg(postprocess));
  _fps_result->setText(message);
  setWindowTitle(message);
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::show_memory_usage(gpucast::beziersurfaceobject::memory_usage const& usage)
{
  auto surface_info = _glwindow->surfaces_total_and_average_degree();
  auto curve_info = _glwindow->curves_total_and_average_degree();

  QString message = "";
  message.append(QString("Surfaces                   : %1 \n").arg(surface_info.first));
  message.append(QString("Trimming Curves            : %1 \n").arg(curve_info.first));
  message.append(QString("Surfaces Average Degree    : %1 \n").arg(surface_info.second));
  message.append(QString("Trimming Curves Avg. Deg   : %1 \n").arg(curve_info.second));
  message.append(QString("Control point data         : %1 kb\n").arg(usage.surface_control_point_data / 1024));
  message.append(QString("Trim point data            : %1 kb\n").arg(usage.trimcurve_control_point_data / 1024));

  message.append(QString("Proxy geometry raycasting  : %1 kb\n").arg(usage.vertex_array_raycasting / 1024));
  message.append(QString("Proxy geometry tesselation : %1 kb\n").arg(usage.vertex_array_tesselation / 1024));

  message.append(QString("Domain partition :\n"));
  message.append(QString(" - contour kd-tree         : %1 kb\n").arg(usage.domain_partition_kd_tree / 1024));
  message.append(QString(" - contour binary          : %1 kb\n").arg(usage.domain_partition_contour_binary / 1024));
  message.append(QString(" - double binary           : %1 kb\n").arg(usage.domain_partition_double_binary / 1024));
  message.append(QString(" - loop lists              : %1 kb\n").arg(usage.domain_partition_loops / 1024));
  _memory_usage->setText(message);
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::update_count(unsigned triangles, unsigned fragments, unsigned culled_triangles, unsigned trimmed_fragments, unsigned estimate)
{
  QString message = QString("Triangles %1\n").arg(triangles);
  message.append(QString("Fragments : %1\n").arg(fragments));
  message.append(QString("Culled triangles : %1\n").arg(culled_triangles));
  message.append(QString("Trimmed fragments : %1\n").arg(trimmed_fragments));
  message.append(QString("Estimate : %1").arg(estimate));
  _counting_result->setText(message);
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
  _glwindow->trimming(find_by_second(_trimming_modes,
    _combobox_trimming->currentText().toStdString(),
    gpucast::beziersurfaceobject::no_trimming));
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::antialiasing()
{
  _glwindow->antialiasing(find_by_second(_antialiasing_modes,
    _combobox_antialiasing->currentText().toStdString(), 
    gpucast::gl::bezierobject::disabled));
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::fillmode()
{
  _glwindow->fillmode(find_by_second(_fill_modes,
    _combobox_fillmode->currentText().toStdString(),
    gpucast::gl::bezierobject::solid));
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::preclassification()
{
  auto pre = find_by_second(_preclassification_modes,
    _combobox_preclassification->currentText().toStdString(),
    gpucast::beziersurfaceobject::trim_preclassification_default_resolution);
  _glwindow->preclassification(pre);
}


///////////////////////////////////////////////////////////////////////////////
void mainwindow::rendering()
{
  _glwindow->rendermode(find_by_second(_rendering_modes,
    _combobox_rendering->currentText().toStdString(), 
    gpucast::gl::bezierobject::raycasting));
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
  connect(_checkbox_culling,                SIGNAL(stateChanged(int)),   _glwindow,    SLOT( backface_culling(int)));
  connect(_checkbox_counting,               SIGNAL(stateChanged(int)),   _glwindow,    SLOT( enable_counter(int)));
  connect(_checkbox_tritesselation,         SIGNAL(stateChanged(int)), _glwindow, SLOT(enable_triangular_tesselation(int)));
  connect(_checkbox_holefilling,            SIGNAL(stateChanged(int)), _glwindow, SLOT(holefilling(int)));
  connect(_checkbox_conservative_rasterization, SIGNAL(stateChanged(int)), _glwindow, SLOT(conservative_rasterization(int)));
  

  connect(_combobox_antialiasing,           SIGNAL(currentIndexChanged(int)), this, SLOT(antialiasing()));
  connect(_combobox_trimming,               SIGNAL(currentIndexChanged(int)), this, SLOT(trimming()));
  connect(_combobox_rendering,              SIGNAL(currentIndexChanged(int)), this, SLOT(rendering()));
  connect(_combobox_fillmode,               SIGNAL(currentIndexChanged(int)), this, SLOT(fillmode()));
  connect(_combobox_preclassification,      SIGNAL(currentIndexChanged(int)), this, SLOT(preclassification()));
  
  connect(_slider_trim_max_bisections,          SIGNAL(valueChanged(int)),    _glwindow, SLOT(trim_max_bisections(int)));
  connect(_slider_trim_error_tolerance,         SIGNAL(valueChanged(float)),  _glwindow, SLOT(trim_error_tolerance(float)));
  connect(_slider_tesselation_max_pixel_error,  SIGNAL(valueChanged(float)),  _glwindow, SLOT(tesselation_max_pixel_error(float)));
  connect(_slider_tesselation_max_object_error, SIGNAL(valueChanged(float)),  _glwindow, SLOT(tesselation_max_geometric_error(float)));
  connect(_slider_raycasting_max_iterations,    SIGNAL(valueChanged(int)),    _glwindow, SLOT(raycasting_max_iterations(int)));
  connect(_slider_raycasting_error_tolerance,   SIGNAL(valueChanged(float)),  _glwindow, SLOT(raycasting_error_tolerance(float)));

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

  auto widget = new QWidget { _menu };
  _menu->setWidget(widget);
  _menu->setFixedWidth(500);
  //_menu->setFixedHeight(1400);
  this->addDockWidget(Qt::RightDockWidgetArea, _menu);

  auto layout = new QVBoxLayout;
  layout->setAlignment(Qt::AlignTop);
  widget->setLayout(layout);

  _counting_result = new QLabel("", _menu);
  _fps_result = new QLabel("", _menu);
  _memory_usage = new QLabel("", _menu);

  // init buttons
  _button_recompile      = new QPushButton("Recompile Shaders", _menu);
  _button_set_spheremap  = new QPushButton("Choose Spheremap", _menu);
  _button_set_diffusemap = new QPushButton("Choose Diffusemap", _menu);

  // init check boxes
  _checkbox_spheremap    = new QCheckBox("Enable Spheremapping", _menu);
  _checkbox_diffusemap   = new QCheckBox("Enable Diffusemapping", _menu);
  _checkbox_fxaa         = new QCheckBox("Enable screen-space FXAA", _menu);
  _checkbox_vsync        = new QCheckBox("Enable VSync", _menu);
  _checkbox_sao          = new QCheckBox("Enable Ambient Occlusion", _menu);
  _checkbox_culling      = new QCheckBox("Backface Culling", _menu);
  _checkbox_counting     = new QCheckBox("Enable Triangle/Fragment Counter", _menu);
  _checkbox_tritesselation = new QCheckBox("Enable Triangular Tesselation", _menu);
  _checkbox_holefilling = new QCheckBox("Enable Holefilling", _menu);
  _checkbox_conservative_rasterization = new QCheckBox("Enable Conservative Rasterization", _menu);

  // init combo boxes
  _combobox_rendering = new QComboBox;
  _combobox_antialiasing = new QComboBox;
  _combobox_trimming     = new QComboBox;
  _combobox_fillmode = new QComboBox;
  _combobox_preclassification = new QComboBox;

  std::for_each(_rendering_modes.begin(), _rendering_modes.end(), [&](std::map<gpucast::gl::bezierobject::render_mode, std::string>::value_type const& p) { _combobox_rendering->addItem(p.second.c_str()); });
  std::for_each(_antialiasing_modes.begin(), _antialiasing_modes.end(), [&](std::map<gpucast::gl::bezierobject::anti_aliasing_mode, std::string>::value_type const& p) { _combobox_antialiasing->addItem(p.second.c_str()); });
  std::for_each(_trimming_modes.begin(), _trimming_modes.end(), [&](std::map<gpucast::beziersurfaceobject::trim_approach_t, std::string>::value_type const& p) { _combobox_trimming->addItem(p.second.c_str()); });
  std::for_each(_fill_modes.begin(), _fill_modes.end(), [&](std::map<gpucast::gl::bezierobject::fill_mode, std::string>::value_type const& p) { _combobox_fillmode->addItem(p.second.c_str()); });
  std::for_each(_preclassification_modes.begin(), _preclassification_modes.end(), [&](std::map<unsigned, std::string>::value_type const& p) { _combobox_preclassification->addItem(p.second.c_str()); });

  _combobox_rendering->setCurrentText(tr(_rendering_modes[gpucast::gl::bezierobject::tesselation].c_str()));
  _combobox_trimming->setCurrentText(tr(_trimming_modes[gpucast::beziersurfaceobject::contour_kd_partition].c_str()));
  _combobox_preclassification->setCurrentText(tr(_preclassification_modes[8].c_str()));

  _slider_trim_max_bisections = new SlidersGroup(Qt::Horizontal, tr("max. bisections"), 1, 32, this);
  _slider_trim_error_tolerance = new FloatSlidersGroup(Qt::Horizontal, tr("max. error tolerance"), 0.0001f, 0.1f,  this);
  _slider_tesselation_max_object_error = new FloatSlidersGroup(Qt::Horizontal, tr("max. object error"), 0.0001f, 1.0f, this);
  _slider_tesselation_max_pixel_error = new FloatSlidersGroup(Qt::Horizontal,tr("max. pixel error"), 0.5f, 64.0f,  this);
  _slider_raycasting_max_iterations = new SlidersGroup(Qt::Horizontal,tr("max. newton iterations"),  1, 16, this);
  _slider_raycasting_error_tolerance = new FloatSlidersGroup(Qt::Horizontal, tr("max. raycasting error"), 0.0001f, 0.1f,  this);

  // apply default values
  _slider_trim_max_bisections->setValue(gpucast::gl::bezierobject::default_render_configuration::trimming_max_bisections);
  _slider_trim_error_tolerance->setValue(gpucast::gl::bezierobject::default_render_configuration::trimming_error_tolerance);
  _slider_tesselation_max_object_error->setValue(gpucast::gl::bezierobject::default_render_configuration::tesselation_max_geometric_error);
  _slider_tesselation_max_pixel_error->setValue(gpucast::gl::bezierobject::default_render_configuration::tesselation_max_pixel_error);
  _slider_raycasting_max_iterations->setValue(gpucast::gl::bezierobject::default_render_configuration::raycasting_max_iterations);
  _slider_raycasting_error_tolerance->setValue(gpucast::gl::bezierobject::default_render_configuration::raycasting_error_tolerance);

  // apply widget into layout
  QGroupBox* system_desc = new QGroupBox("System", this);
  QVBoxLayout* system_desc_layout = new QVBoxLayout;
  system_desc_layout->addWidget(_button_recompile);
  system_desc_layout->addWidget(_checkbox_vsync);
  system_desc_layout->addWidget(_checkbox_counting);
  system_desc_layout->addWidget(_counting_result);
  //system_desc_layout->addWidget(_fps_result);
  system_desc_layout->addWidget(_memory_usage);
  system_desc->setLayout(system_desc_layout);
  layout->addWidget(system_desc);

  QGroupBox* aa_desc = new QGroupBox("Anti-Aliasing", this);
  QVBoxLayout* aa_desc_layout = new QVBoxLayout;
  aa_desc_layout->addWidget(_checkbox_fxaa);
  aa_desc_layout->addWidget(_checkbox_holefilling);
  aa_desc_layout->addWidget(new QLabel("Shader-based anti-aliasing"));
  aa_desc_layout->addWidget(_combobox_antialiasing);
  aa_desc->setLayout(aa_desc_layout);
  layout->addWidget(aa_desc);

  QGroupBox* trim_desc = new QGroupBox("Trimming", this);
  QVBoxLayout* trim_desc_layout = new QVBoxLayout;
  trim_desc_layout->addWidget(_combobox_trimming);
  trim_desc_layout->addWidget(_combobox_preclassification);
  trim_desc_layout->addWidget(_slider_trim_max_bisections);
  trim_desc_layout->addWidget(_slider_trim_error_tolerance);
  trim_desc->setLayout(trim_desc_layout);
  layout->addWidget(trim_desc);

  QGroupBox* rendering_desc = new QGroupBox("Rendering", this);
  QVBoxLayout* rendering_desc_layout = new QVBoxLayout;
  rendering_desc_layout->addWidget(_combobox_rendering);
  rendering_desc_layout->addWidget(_combobox_fillmode);
  rendering_desc_layout->addWidget(_checkbox_culling);
  rendering_desc_layout->addWidget(_checkbox_tritesselation);
  rendering_desc_layout->addWidget(_checkbox_conservative_rasterization);
  rendering_desc_layout->addWidget(_slider_tesselation_max_pixel_error);
  rendering_desc_layout->addWidget(_slider_tesselation_max_object_error);
  rendering_desc_layout->addWidget(_slider_raycasting_max_iterations);
  rendering_desc_layout->addWidget(_slider_raycasting_error_tolerance);
  rendering_desc->setLayout(rendering_desc_layout);
  layout->addWidget(rendering_desc);

  QGroupBox* shading_desc = new QGroupBox("Shading", this);
  QVBoxLayout* shading_desc_layout = new QVBoxLayout;
  shading_desc_layout->addWidget(_checkbox_spheremap);
  shading_desc_layout->addWidget(_button_set_spheremap);
  shading_desc_layout->addWidget(_checkbox_diffusemap);
  shading_desc_layout->addWidget(_button_set_diffusemap);
  shading_desc_layout->addWidget(_checkbox_sao);
  shading_desc->setLayout(shading_desc_layout);
  layout->addWidget(shading_desc);
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_statusbar()
{}





