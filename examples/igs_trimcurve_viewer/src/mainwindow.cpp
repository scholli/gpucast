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
  _recompile_button = new QPushButton;
  _recompile_button->setText("Recompile Shaders");
  _show_texel_fetches = new QCheckBox;

  QGridLayout* layout = new QGridLayout(this);
  _controlwidget->setLayout(layout);

  _list_object->setSelectionMode(QAbstractItemView::SingleSelection);
  _list_surface->setSelectionMode(QAbstractItemView::SingleSelection);

  layout->addWidget(_list_object);
  layout->addWidget(_list_surface);
  layout->addWidget(_viewbox);
  layout->addWidget(_recompile_button);

  layout->addWidget(_show_texel_fetches);
  _show_texel_fetches->setText("Show texel fetches");

  layout->addWidget(_label_mem);
  layout->addWidget(_label_fps);

  /////////////////////////////////////
  // file menu
  /////////////////////////////////////
  _file_menu = menuBar()->addMenu(tr("File"));
  _action_loadfile = new QAction(tr("Open"), _controlwidget);
  _file_menu->addAction(_action_loadfile);

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

  _modes.insert(std::make_pair(glwidget::minification, "minification"));

  std::for_each ( _modes.begin(), _modes.end(), [&] ( std::map<glwidget::view, std::string>::value_type const& p ) { _viewbox->addItem(p.second.c_str()); } );

  /////////////////////////////////////
  // actions
  /////////////////////////////////////
  connect(_action_loadfile, SIGNAL(triggered()), this, SLOT(openfile()));
  connect(_list_object, SIGNAL(itemSelectionChanged()), this, SLOT(update_surfacelist()));
  connect(_list_surface, SIGNAL(itemSelectionChanged()), this, SLOT(update_view()));
  connect(_viewbox, SIGNAL(currentIndexChanged(int)), this, SLOT(update_view()));
  connect(_recompile_button, SIGNAL(clicked()), _glwindow, SLOT(recompile()));
  connect(_show_texel_fetches, SIGNAL(stateChanged(int)), this, SLOT(show_texel_fetches()));
  
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
      std::size_t trimcurves = _glwindow->get_domain(objectname, i)->size();
      std::size_t trimloops  = _glwindow->get_domain(objectname, i)->loop_count();
      std::string surfacename = boost::lexical_cast<std::string>(i) + " (" 
        + boost::lexical_cast<std::string>(trimloops) + " loops, " + boost::lexical_cast<std::string>(trimcurves) + " curves)";
      _list_surface->addItem ( surfacename.c_str() );
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::update_view () const 
{
  auto objects  = _list_object->selectedItems();
  auto surfaces = _list_surface->selectedItems();

  if ( !objects.empty() )
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

    _glwindow->update_view ( objects.front()->text().toStdString(), id, current_view);
  }
}

///////////////////////////////////////////////////////////////////////////////
void                    
mainwindow::show_texel_fetches()
{
  _glwindow->show_texel_fetches(_show_texel_fetches->isChecked());
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
