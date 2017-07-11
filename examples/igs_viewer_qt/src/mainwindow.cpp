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
#include <QtWidgets/QListWidget>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QColorDialog>
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
  _create_widgets(argc, argv);
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
  _menu->update();
  _shading_menu->update();

  trimming();

  set_button_color(_current_ambient, _current_material.ambient);
  set_button_color(_current_diffuse, _current_material.diffuse);
  set_button_color(_current_specular, _current_material.specular);

  std::cout << _current_material.ambient.redF() << _current_material.ambient.greenF() << _current_material.ambient.blueF() << std::endl;
  std::cout << _current_material.diffuse.redF() << _current_material.diffuse.greenF() << _current_material.diffuse.blueF() << std::endl;
  std::cout << _current_material.specular.redF() << _current_material.specular.greenF() << _current_material.specular.blueF() << std::endl;

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
void mainwindow::set_button_color(QPushButton* button, QColor const& color)
{
  QString qss = QString("background-color: %1").arg(color.name());
  button->setStyleSheet(qss);
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::openfile()
{
 QStringList qfilelist = QFileDialog::getOpenFileNames(this, tr("Open Volume"), ".", tr("Surface Files (*.igs *.cfg *.dat)"));
 std::list<std::string> filelist;
 std::transform(qfilelist.begin(), qfilelist.end(), std::back_inserter(filelist), []( QString const& qstr ) { return qstr.toStdString(); } );
 _glwindow->open(filelist); 

 _object_list->addItems(qfilelist);

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

 _object_list->addItems(qfilelist);

 update_interface();
}

///////////////////////////////////////////////////////////////////////////////
void
mainwindow::deletefiles()
{
  std::list<std::string> filelist;

  auto items = _object_list->selectedItems();

  for (auto item : items) {
    filelist.push_back(item->text().toStdString());
  }

  _glwindow->remove(filelist);

  qDeleteAll(_object_list->selectedItems());

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
void mainwindow::set_specular()
{
  QColor color = QColorDialog::getColor(_current_material.specular, this);
  std::cout << color.greenF() << std::endl;
  _current_material.specular = color;
  update_interface();
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::set_diffuse()
{
  QColor color = QColorDialog::getColor(_current_material.diffuse, this);
  std::cout << color.greenF() << std::endl;
  _current_material.diffuse = color;
  update_interface();
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::set_ambient()
{
  QColor color = QColorDialog::getColor(_current_material.ambient, this);
  std::cout << color.greenF() << std::endl;
  _current_material.ambient = color;
  update_interface();
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::set_shininess(float s)
{
  _current_material.shininess = s;
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::set_opacity(float o)
{
  _current_material.opacity = o;
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::apply_material()
{ 
  auto items = _object_list->selectedItems();

  gpucast::math::vec4f mat_amb(_current_material.ambient.redF(), _current_material.ambient.greenF(), _current_material.ambient.blueF(), 1.0);
  gpucast::math::vec4f mat_diff(_current_material.diffuse.redF(), _current_material.diffuse.greenF(), _current_material.diffuse.blueF(), 1.0);
  gpucast::math::vec4f mat_spec(_current_material.specular.redF(), _current_material.specular.greenF(), _current_material.specular.blueF(), 1.0);

  for (auto item : items) {

    auto name = item->text().toStdString();

    _glwindow->apply_material(name, mat_amb, mat_diff, mat_spec, _current_material.shininess, _current_material.opacity);
  }

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

  connect(_addfile_button, SIGNAL(released()), this, SLOT(addfile()));
  connect(_deletefile_button, SIGNAL(released()), this, SLOT(deletefiles()));

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

  connect(_current_specular, SIGNAL(released()), this, SLOT(set_specular()));
  connect(_current_diffuse, SIGNAL(released()), this, SLOT(set_diffuse()));
  connect(_current_ambient, SIGNAL(released()), this, SLOT(set_ambient()));
  connect(_current_shininess, SIGNAL(valueChanged(float)), this, SLOT(set_shininess(float)));
  connect(_current_opacity, SIGNAL(valueChanged(float)), this, SLOT(set_opacity(float)));

  connect(_material_apply, SIGNAL(released()), this, SLOT(apply_material()));

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
  _glwindow->resize(_width, _height);
  resize(_width, _height);
  setCentralWidget(_glwindow);
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_menus()
{
  _shading_menu = new QMainWindow();
  auto shading_layout = new QVBoxLayout;
  shading_layout->setAlignment(Qt::AlignTop);

  auto label_color_a = new QLabel("Ambient", _shading_menu);
  auto label_color_d = new QLabel("Diffuse", _shading_menu);
  auto label_color_s = new QLabel("Specular", _shading_menu);
  auto label_color_sh = new QLabel("Shininess", _shading_menu);
  auto label_color_o = new QLabel("Opacity", _shading_menu);

  _addfile_button = new QPushButton("Add");
  _deletefile_button = new QPushButton("Delete");
  _material_apply = new QPushButton("Apply");

  _current_specular = new QPushButton(" ");
  _current_diffuse = new QPushButton(" ");
  _current_ambient = new QPushButton(" ");
  _current_shininess = new FloatSlidersGroup(Qt::Horizontal, tr("Shininess"), 0.0, 1.0f, _shading_menu);
  _current_shininess->setValue(0.0);
  _current_opacity = new FloatSlidersGroup(Qt::Horizontal, tr("Opacity"), 0.0, 1.0f, _shading_menu);
  _current_opacity->setValue(1.0);
  
  _current_specular->setFixedSize(40, 40);
  _current_diffuse->setFixedSize(40, 40);
  _current_ambient->setFixedSize(40, 40);

  shading_layout->addWidget(_addfile_button);
  shading_layout->addWidget(_deletefile_button);

  auto gridlayout = new QGridLayout();
  auto color_widget = new QWidget();
  color_widget->setLayout(gridlayout);

  gridlayout->addWidget(label_color_s, 0, 0);
  gridlayout->addWidget(_current_specular, 0, 1);

  gridlayout->addWidget(label_color_d, 1, 0);
  gridlayout->addWidget(_current_diffuse, 1, 1);

  gridlayout->addWidget(label_color_a, 2, 0);
  gridlayout->addWidget(_current_ambient, 2, 1);

  _object_list = new QListWidget(_shading_menu);
  _object_list->setSelectionMode(QAbstractItemView::MultiSelection);

  shading_layout->addWidget(_object_list);
  
  shading_layout->addWidget(color_widget);
  shading_layout->addWidget(_current_shininess);
  shading_layout->addWidget(_current_opacity);

  shading_layout->addWidget(_material_apply);
  //////////////////////////////////////////////////


  _menu = new QMainWindow();
  auto layout = new QVBoxLayout;

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

  _slider_trim_max_bisections = new SlidersGroup(Qt::Horizontal, tr("max. bisections"), 1, 32, _menu);
  _slider_trim_error_tolerance = new FloatSlidersGroup(Qt::Horizontal, tr("max. error tolerance"), 0.0001f, 0.1f, _menu);
  _slider_tesselation_max_object_error = new FloatSlidersGroup(Qt::Horizontal, tr("max. object error"), 0.0001f, 1.0f, _menu);
  _slider_tesselation_max_pixel_error = new FloatSlidersGroup(Qt::Horizontal,tr("max. pixel error"), 0.5f, 64.0f, _menu);
  _slider_raycasting_max_iterations = new SlidersGroup(Qt::Horizontal,tr("max. newton iterations"),  1, 16, _menu);
  _slider_raycasting_error_tolerance = new FloatSlidersGroup(Qt::Horizontal, tr("max. raycasting error"), 0.0001f, 0.1f, _menu);

  // apply default values
  _slider_trim_max_bisections->setValue(gpucast::gl::bezierobject::default_render_configuration::trimming_max_bisections);
  _slider_trim_error_tolerance->setValue(gpucast::gl::bezierobject::default_render_configuration::trimming_error_tolerance);
  _slider_tesselation_max_object_error->setValue(gpucast::gl::bezierobject::default_render_configuration::tesselation_max_geometric_error);
  _slider_tesselation_max_pixel_error->setValue(gpucast::gl::bezierobject::default_render_configuration::tesselation_max_pixel_error);
  _slider_raycasting_max_iterations->setValue(gpucast::gl::bezierobject::default_render_configuration::raycasting_max_iterations);
  _slider_raycasting_error_tolerance->setValue(gpucast::gl::bezierobject::default_render_configuration::raycasting_error_tolerance);

  // apply widget into layout
  QGroupBox* system_desc = new QGroupBox("System", _menu);
  QVBoxLayout* system_desc_layout = new QVBoxLayout;
  system_desc_layout->addWidget(_button_recompile);
  system_desc_layout->addWidget(_checkbox_vsync);
  system_desc_layout->addWidget(_checkbox_counting);
  system_desc_layout->addWidget(_counting_result);
  //system_desc_layout->addWidget(_fps_result);
  system_desc_layout->addWidget(_memory_usage);
  system_desc->setLayout(system_desc_layout);
  layout->addWidget(system_desc);

  QGroupBox* aa_desc = new QGroupBox("Anti-Aliasing", _menu);
  QVBoxLayout* aa_desc_layout = new QVBoxLayout;
  aa_desc_layout->addWidget(_checkbox_fxaa);
  aa_desc_layout->addWidget(_checkbox_holefilling);
  aa_desc_layout->addWidget(new QLabel("Shader-based anti-aliasing"));
  aa_desc_layout->addWidget(_combobox_antialiasing);
  aa_desc->setLayout(aa_desc_layout);
  layout->addWidget(aa_desc);

  QGroupBox* trim_desc = new QGroupBox("Trimming", _menu);
  QVBoxLayout* trim_desc_layout = new QVBoxLayout;
  trim_desc_layout->addWidget(_combobox_trimming);
  trim_desc_layout->addWidget(_combobox_preclassification);
  trim_desc_layout->addWidget(_slider_trim_max_bisections);
  trim_desc_layout->addWidget(_slider_trim_error_tolerance);
  trim_desc->setLayout(trim_desc_layout);
  layout->addWidget(trim_desc);

  QGroupBox* rendering_desc = new QGroupBox("Rendering", _menu);
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

  QGroupBox* shading_desc = new QGroupBox("Shading", _menu);
  QVBoxLayout* shading_desc_layout = new QVBoxLayout;
  shading_desc_layout->addWidget(_checkbox_spheremap);
  shading_desc_layout->addWidget(_button_set_spheremap);
  shading_desc_layout->addWidget(_checkbox_diffusemap);
  shading_desc_layout->addWidget(_button_set_diffusemap);
  shading_desc_layout->addWidget(_checkbox_sao);
  shading_desc->setLayout(shading_desc_layout);
  layout->addWidget(shading_desc);

  layout->setAlignment(Qt::AlignTop);

  auto param_body = new QWidget ();
  param_body->setFixedWidth(800);
  param_body->setLayout(layout);
  _menu->setCentralWidget(param_body);
  _menu->show();

  auto shading_body = new QWidget();
  shading_body->setLayout(shading_layout);
  shading_body->setFixedWidth(800);
  _shading_menu->setCentralWidget(shading_body);
  _shading_menu->show();

  _file_menu = _shading_menu->menuBar()->addMenu(tr("File"));
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_statusbar()
{}





