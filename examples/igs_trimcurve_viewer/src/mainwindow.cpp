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

#include <QtWidgets/QMenuBar>
#include <QSettings>
#include <QtWidgets/QGridLayout>

// system includes
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>


///////////////////////////////////////////////////////////////////////////////
mainwindow::mainwindow( int argc, char** argv, unsigned width, unsigned height )
: _width                              ( width ), 
  _height                             ( height )
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
  _glwindow->setFixedHeight(_height);
  _glwindow->setFixedWidth(_width);
  this->setCentralWidget(_glwindow);

  /////////////////////////////////////
  // control widget
  /////////////////////////////////////
  _controlwidget    = new QWidget;
  _list_object      = new QListWidget;
  _list_surface     = new QListWidget;
  _viewbox          = new QComboBox;
  _label_fps        = new QLabel;
  _label_mem        = new QLabel;
  _label_size       = new QLabel;

  _recompile_button = new QPushButton;
  _resetview_button = new QPushButton;
  _resetview_button->setText("Reset View");
  _recompile_button->setText("Recompile");

  _show_texel_fetches = new QCheckBox;
  _linear_texture_filter = new QCheckBox;
  _optimal_distance   = new QCheckBox;
  _antialiasing       = new QComboBox;

  _zoom_control = new SlidersGroup(Qt::Horizontal, tr("Zoom"), this);
  _zoom_control->setMinimum(0);
  _zoom_control->setMaximum(_width);
  _zoom_control->setValue(0);

  _texture_resolution = new QComboBox;
  std::vector<unsigned> texture_resolutions = { 8, 16, 32, 64, 128 };
  for (auto i = 0; i != texture_resolutions.size(); ++i) {
    _texture_resolution->insertItem(i, tr(std::to_string(texture_resolutions[i]).c_str()), texture_resolutions[i]);
  }

  _pixel_size = new QComboBox;
  std::vector<unsigned> pixel_sizes = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
  for (auto i = 0; i != pixel_sizes.size(); ++i) { 
    _pixel_size->insertItem(i, tr(std::to_string(pixel_sizes[i]).c_str()), pixel_sizes[i]);
  }

  QGridLayout* layout = new QGridLayout;
  _controlwidget->setLayout(layout);

  _list_object->setSelectionMode(QAbstractItemView::SingleSelection);
  _list_surface->setSelectionMode(QAbstractItemView::SingleSelection);

  unsigned row = 0;

  layout->setAlignment(Qt::AlignRight);

  layout->addWidget(_list_object, row, 0);
  layout->addWidget(_list_surface, row++, 1);

  layout->addWidget(new QLabel("View Mode:"), row, 0);
  layout->addWidget(_viewbox, row++, 1);

  layout->addWidget(new QLabel("Compile Shaders:"), row, 0);
  layout->addWidget(_recompile_button, row++, 1);

  layout->addWidget(new QLabel("Reset View:"), row, 0);
  layout->addWidget(_resetview_button, row++, 1);

  layout->addWidget(new QLabel("Texture Resolution:"), row, 0);
  layout->addWidget(_texture_resolution, row++, 1);

  layout->addWidget(new QLabel("Pixel Size:"), row, 0);
  layout->addWidget(_pixel_size, row++, 1);

  layout->addWidget(_show_texel_fetches, row++, 1);
  _show_texel_fetches->setText("Show Texel Fetches");

  layout->addWidget(_linear_texture_filter, row++, 1);
  _linear_texture_filter->setText("Linear Texture Filtering");

  layout->addWidget(_optimal_distance, row++, 1);
  _optimal_distance->setText("Optimal Distance Field");

  layout->addWidget(_antialiasing, row++, 1);
  layout->addWidget(_zoom_control, row++, 1);
  
  layout->addWidget(_label_mem, row++, 0);
  layout->addWidget(_label_fps, row++, 0);
  layout->addWidget(_label_size, row++, 0);

  /////////////////////////////////////
  // file menu
  /////////////////////////////////////
  _file_menu = menuBar()->addMenu(tr("File"));

  _action_loadfile = new QAction(tr("Open"), _glwindow);
  _file_menu->addAction(_action_loadfile);

  /////////////////////////////////////
  // anti-aliasing modes
  /////////////////////////////////////
  _aamodes.insert(std::make_pair(glwidget::disabled, "No Anti-Aliasing"));
  _aamodes.insert(std::make_pair(glwidget::prefiltered_edge_estimation, "Prefiltered Edge Estimation"));
  _aamodes.insert(std::make_pair(glwidget::supersampling2x2, "Supersampling(2x2)"));
  _aamodes.insert(std::make_pair(glwidget::supersampling3x3, "Supersampling(3x3)"));
  _aamodes.insert(std::make_pair(glwidget::supersampling4x4, "Supersampling(4x4)"));
  _aamodes.insert(std::make_pair(glwidget::supersampling8x8, "Supersampling(8x8)"));

  std::for_each(_aamodes.begin(), _aamodes.end(), [&](std::map<glwidget::aamode, std::string>::value_type const& p) { _antialiasing->addItem(p.second.c_str()); });

  /////////////////////////////////////
  // view modes
  /////////////////////////////////////
  _modes.insert(std::make_pair(glwidget::original, "original"));

  _modes.insert(std::make_pair(glwidget::double_binary_partition, "double_binary_partition"));
  _modes.insert(std::make_pair(glwidget::double_binary_classification, "double_binary_classification"));

  _modes.insert(std::make_pair(glwidget::contour_map_binary_partition, "contour_map_binary_partition"));
  _modes.insert(std::make_pair(glwidget::contour_map_binary_classification, "contour_map_binary_classification"));

  _modes.insert(std::make_pair(glwidget::contour_map_loop_list_partition, "contour_map_loop_list_partition"));
  _modes.insert(std::make_pair(glwidget::contour_map_loop_list_classification, "contour_map_loop_list_classification"));

  _modes.insert(std::make_pair(glwidget::minification, "sampling"));

  _modes.insert(std::make_pair(glwidget::binary_field, "binary_texture"));
  _modes.insert(std::make_pair(glwidget::distance_field, "distance_field_texture"));

  _modes.insert(std::make_pair(glwidget::prefilter, "prefilter"));

  std::for_each ( _modes.begin(), _modes.end(), [&] ( std::map<glwidget::view, std::string>::value_type const& p ) { _viewbox->addItem(p.second.c_str()); } );

  /////////////////////////////////////
  // actions
  /////////////////////////////////////
  connect(_action_loadfile, SIGNAL(triggered()), this, SLOT(openfile()));
  connect(_list_object, SIGNAL(itemSelectionChanged()), this, SLOT(update_surfacelist()));
  connect(_list_surface, SIGNAL(itemSelectionChanged()), this, SLOT(update_view()));
  connect(_viewbox, SIGNAL(currentIndexChanged(int)), this, SLOT(update_view()));
  connect(_recompile_button, SIGNAL(clicked()), _glwindow, SLOT(recompile()));
  connect(_resetview_button, SIGNAL(clicked()), _glwindow, SLOT(resetview()));
  connect(_show_texel_fetches, SIGNAL(stateChanged(int)), _glwindow, SLOT(show_texel_fetches(int)));
  connect(_linear_texture_filter, SIGNAL(stateChanged(int)), _glwindow, SLOT(texture_filtering(int)));
  connect(_texture_resolution, SIGNAL(currentIndexChanged(int)), SLOT(update_view()));
  connect(_optimal_distance, SIGNAL(stateChanged(int)), _glwindow, SLOT(optimal_distance(int)));
  connect(_pixel_size, SIGNAL(currentIndexChanged(int)), this, SLOT(pixel_size_changed()));
  connect(_antialiasing, SIGNAL(currentIndexChanged(int)), this, SLOT(antialiasing()));
  connect(_zoom_control, SIGNAL(valueChanged(int)), this, SLOT(zoom_changed(int)));

  _controlwidget->show();
  this->show();

  QMainWindow::update();
}


///////////////////////////////////////////////////////////////////////////////
mainwindow::~mainwindow()
{}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::update_objectlist ()
{
  _list_object->clear();
  _list_surface->clear();

  for ( std::string const& filename : _filenames )
  {
    _list_object->addItem(filename.c_str());  
  }
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::update_surfacelist ()
{
  _list_surface->clear();

  QList<QListWidgetItem*> items = _list_object->selectedItems();
  
  if ( !items.empty() ) // only first item
  {
    std::string objectname = items.front()->text().toStdString();
    std::size_t nsurfaces = _glwindow->get_surfaces(objectname);
    for ( std::size_t i = 0; i != nsurfaces; ++i )
    {
      std::size_t trimcurves  = _glwindow->get_domain(objectname, i)->size();
      std::size_t trimloops   = _glwindow->get_domain(objectname, i)->loop_count();
      std::size_t max_degree  = _glwindow->get_domain(objectname, i)->max_degree();
      std::string surfacename = boost::lexical_cast<std::string>(i) + " (" 
        + boost::lexical_cast<std::string>(trimloops)+" loops, " + boost::lexical_cast<std::string>(trimcurves)+" curves), max. degree: " + boost::lexical_cast<std::string>(max_degree);
      _list_surface->addItem ( surfacename.c_str() );
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::update_view () const 
{
  auto objects  = _list_object->selectedItems();
  auto surfaces = _list_surface->selectedItems();

  if (!objects.empty() && !surfaces.empty())
  {
    std::string surface_str = surfaces.front()->text().toStdString();
    std::stringstream sstr ( surface_str );
    std::string id_str; 
    sstr >> id_str;
    unsigned id = boost::lexical_cast<unsigned>(id_str.c_str());

    std::string viewstr = _viewbox->currentText().toStdString();
    glwidget::view current_view = glwidget::original;

    for ( auto mode : _modes )
    {
      if ( mode.second == viewstr ) current_view = mode.first;
    }

    _glwindow->update_view(objects.front()->text().toStdString(), id, current_view, _texture_resolution->itemData(_texture_resolution->currentIndex()).toUInt() );
  }
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::antialiasing() const
{
  for (auto const& mode : _aamodes) {
    if (mode.second == _antialiasing->currentText().toStdString()) {
      _glwindow->antialiasing(mode.first);
    }
  }

}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::pixel_size_changed() const
{
  _glwindow->pixel_size(_pixel_size->itemData(_pixel_size->currentIndex()).toUInt());
}

///////////////////////////////////////////////////////////////////////////////
void                    
mainwindow::zoom_changed(int value)
{
  auto normalized_zoom = float(value) / (float(_zoom_control->getMaximum()) - float(_zoom_control->getMinimum()));
  _glwindow->zoom(normalized_zoom);
}

///////////////////////////////////////////////////////////////////////////////
void                    
mainwindow::show_drawtime ( double ms )
{
  _fps.push_front(ms);

  if ( _fps.size() > 50 ) 
  {
    _fps.pop_back();
  }

  double average_ms = std::accumulate(_fps.begin(), _fps.end(), 0.0 );

  _label_fps->setText("ms / 50 frames: " + QString::number(average_ms) );
}


///////////////////////////////////////////////////////////////////////////////
void                    
mainwindow::show_memusage ( std::size_t bytes ) const
{
  _label_mem->setText("Mem: " + QString::number(bytes) + " Bytes" );
}

///////////////////////////////////////////////////////////////////////////////
void                    
mainwindow::show_domainsize(float umin, float vmin, float umax, float vmax) const {
  _label_size->setText("min: [" + QString::number(umin) + "," + QString::number(vmin) + "], [" + QString::number(umax) + ", " + QString::number(vmax) + "]");
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::closeEvent(QCloseEvent* /*event*/)
{
  _controlwidget->close();
  _glwindow->close();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::openfile()
{
 _filenames.clear();
 QStringList qfilelist = QFileDialog::getOpenFileNames(this, tr("Open IGS File"), ".", tr("Surface Files (*.igs)"));

 std::transform(qfilelist.begin(), qfilelist.end(), std::back_inserter(_filenames), []( QString const& qstr ) { return qstr.toStdString(); } );
 _glwindow->open(_filenames); 

 update_objectlist();
}
