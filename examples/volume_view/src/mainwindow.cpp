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
#include <QCloseEvent>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QStatusBar>  
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QAction>
#include <QtCore/QString>

#include <iostream>

#include <boost/filesystem.hpp>

#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/vsync.hpp>

#include <gpucast/volume/import/xml.hpp>
#include <gpucast/volume/import/xml2.hpp>
#include <gpucast/volume/import/mesh3d.hpp>
#include <gpucast/volume/import/bin.hpp>
#include <gpucast/volume/uid.hpp>

#include <gpucast/volume/volume_converter.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/surface_converter.hpp>


///////////////////////////////////////////////////////////////////////////////
mainwindow::mainwindow( int argc, char** argv, unsigned width, unsigned height )
: _width                              ( width ), 
  _height                             ( height ),
  _interval_min_sample_distance       ( 0.001f, 0.1f ),
  _interval_max_sample_distance       ( 0.001f, 1.0f ),
  _interval_adaptive_sample_scale     ( 0.05f,  1.0f ),
  _interval_max_binary_searches       ( 1, 16 ),
  _interval_epsilon_newton_iteration  ( 0.000001f, 0.01f ),
  _interval_max_newton_iteration      ( 1, 16 ),
  _interval_relative_isovalue         ( 0.0f, 1.0f ),
  _interval_transparency              ( 0.0f, 1.0f ),
  _slider_width                       ( 450 ),
  _current_file_name                  ()
{
  _create_widgets( argc, argv );
  _create_actions();
  _create_menus();
  _create_statusbar();

  apply_default_settings();
  apply_settings_to_interface();

  setUnifiedTitleAndToolBarOnMac(true);

  // generate map with modes and according enums of glwindow
  _modemap.insert         ( std::make_pair ( glwidget::octree_isosurface        , QString("Octreebased Isosurface Raycasting" ) ) );
  _modemap.insert         ( std::make_pair ( glwidget::splatting                , QString("Splatting"                         ) ) );
  _modemap.insert         ( std::make_pair ( glwidget::face_interval_raycasting , QString("Face Interval Raycaster"           ) ) );  
  _modemap.insert         ( std::make_pair ( glwidget::grid_isosurface,           QString("Gridbased Isosurface Raycasting"   ) ) );
  //_modemap.insert         ( std::make_pair ( glwidget::tesselator          , QString("Tesselator"                     ) ) );
  
  // show window
  show();

  // start timing
  _timer.start();

  _box_choose_rendermode->setDisabled(true);
  std::for_each           ( _modemap.begin(), _modemap.end(), [&] ( std::pair<glwidget::rendermode_t, QString> const& p ) { _box_choose_rendermode->addItem( p.second ); } );
  _box_choose_rendermode->setEnabled(true);
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

  bool vsync;
  gpucast::gl::get_vsync(vsync);
  _checkbox_vsync->setChecked(vsync);
}


///////////////////////////////////////////////////////////////////////////////
double 
mainwindow::frame ( )
{
  ++_frames;

  _timer.stop();
  gpucast::gl::time_duration time = _timer.result();
  _timer.start();

  double fps = 1.0 / time.as_seconds();

  if ( _frames > 4 || time.as_seconds() > 1.0 ) // show fps and reset
  {
    _frames = 0;
    show_fps(fps);
  }
  return 1000.0/fps;
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::show_fps ( double fps )
{
  QString message;
  message.append("draw: ");
  message.append(QString("%1").arg(1000.0/fps));
  message.append(" ms, fps: ");
  message.append(QString("%1").arg(fps));
  message.append(" Hz");
  _label_fps->setText(message);
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::open_volume_file()
{
  QString qfilename = QFileDialog::getOpenFileName(this, tr("Open Volume"), ".", tr("Volume Files (*.xml *.dat *.bin)"));

  boost::filesystem::path filename = qfilename.toStdString();

  if ( !boost::filesystem::exists(filename) )
  {
    return;
  } else {
    // determine file extension
    std::string fileextension = boost::filesystem::extension(filename);
    boost::filesystem::path bin_bezierobject_filename = boost::filesystem::change_extension ( filename, ".bin" );

    // reset old objects
    _bezierobject.reset(new gpucast::beziervolumeobject);
    _nurbsobject.reset(new gpucast::nurbsvolumeobject);

    bool read_success = false;

    // try to load from binary
    if ( boost::filesystem::exists ( bin_bezierobject_filename ) )
    {
      gpucast::bin_loader bin_loader;
      read_success = bin_loader.load ( bin_bezierobject_filename.string(), _bezierobject );
      _nurbsobject = _bezierobject->parent();
    } else { /// load ascii data
      if ( fileextension == ".xml" ) 
      {
        gpucast::xml2_loader xml_loader;
        //gpucast::xml_loader xml_loader;
        read_success = xml_loader.load ( filename.string(), _nurbsobject);
      } 

      if ( fileextension == ".dat" ) 
      {
        gpucast::mesh3d_loader mesh_loader;
        read_success = mesh_loader.load ( filename.string(), _nurbsobject );
      }

      // convert to binary and write
      _volume_converter.convert(_nurbsobject, _bezierobject);
      std::fstream ofstr ( bin_bezierobject_filename.string(), std::ios_base::out | std::ios_base::binary );
      _bezierobject->write( ofstr );
      ofstr.close();
    }

    if ( read_success ) {
      _current_file_name = filename.string();
      apply_attributes_to_interface();
    }
  }

  // reset trackball to original position
  _glwindow->reset_trackball();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::open_surface_file()
{
  QStringList  qfilenames = QFileDialog::getOpenFileNames(this, tr("Open Surface"), ".", tr("Surface Files (*.dat)"));

  BOOST_FOREACH ( QString const& qfilename, qfilenames ) 
  {
    boost::filesystem::path filename = qfilename.toStdString();

    if ( !boost::filesystem::exists(filename) )
    {
      return;
    } else {
      // determine file extension
      std::string fileextension = boost::filesystem::extension(filename);

      // create object
      bool read_success = false;

      if ( fileextension == ".dat" ) 
      {
        //std::shared_ptr<gpucast::nurbssurfaceobject> surfobj ( new gpucast::nurbssurfaceobject );
        //
        //gpucast::mesh3d_loader mesh_loader;
        //read_success = mesh_loader.load ( filename.string(), surfobj );
        //gpucast::surface_renderer_gl::drawable_ptr surface = _glwindow->surface_renderer()->create();

        //gpucast::surface_converter converter;
        //converter.convert(surfobj, surface); 

        //gpucast::gl::material random_material ( gpucast::math::vec4f(0.01f, 0.01f, 0.01f, 1.0f),
        //                                 gpucast::math::vec4f(0.5f, 0.5f, 0.5f, 1.0f),
        //                                 gpucast::math::vec4f(0.1f, 0.1f, 0.1f, 1.0f), 0.02f, 1.0f );
        //surface->material(random_material);
        //
        //_glwindow->boundingbox ( glwidget::boundingbox_t ( gpucast::math::point3f(surface->bbox().min), gpucast::math::point3f(surface->bbox().max) ) );
      }
    }
  }

  // reset trackball to original position
  _glwindow->reset_trackball();
}


///////////////////////////////////////////////////////////////////////////////
void
mainwindow::apply_volume_to_renderer()
{
  // set boundingbox for renderer
  if ( _nurbsobject && _bezierobject )
  {
    std::string attribute_name = _box_choose_attribute->currentText().toStdString();

    _glwindow->resize(_width, _height);

    if ( !attribute_name.empty() )
    {
      _glwindow->apply( _settings, 
                        _nurbsobject, 
                        _bezierobject, 
                        attribute_name, 
                        _current_file_name, 
                        _get_mode(_box_choose_rendermode->currentText()));
    }
  }

  update_interface(); 
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_adaptive_sampling ()
{
  _settings.isosearch_adaptive_sampling = _checkbox_adaptive_sampling->checkState() == Qt::Checked;
  apply_settings_to_renderer();
} 


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_show_isosides ()
{
  _settings.isosides = _checkbox_show_isosides->checkState() == Qt::Checked;
  apply_settings_to_renderer();
} 


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_fxaa ()
{
  _settings.fxaa = _checkbox_fxaa->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_vsync ()
{
  bool enable_vsync = _checkbox_vsync->checkState() == Qt::Checked;
  gpucast::gl::set_vsync ( enable_vsync );
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_backface_culling ()
{
  _settings.cullface = _checkbox_backface_culling->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_show_face_samples ()
{
  _settings.visualization_props.show_samples_face_intersection = _checkbox_show_face_samples->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_show_face_intersections ()
{
  _settings.visualization_props.show_face_intersections = _checkbox_show_face_intersections->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_show_face_intersection_tests ()
{
  _settings.visualization_props.show_face_intersection_tests = _checkbox_show_face_intersection_tests->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_newton_inflection ()
{
  _settings.detect_implicit_inflection = _checkbox_newton_inflection->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_newton_extremum ()
{
  _settings.detect_implicit_extremum = _checkbox_newton_extremum->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_sample_based_face_intersection ()
{
  _settings.sample_based_face_intersection = _checkbox_sample_based_face_intersection->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_min_sample_distance ()
{
  float relative = float(_slider_min_sample_distance->sliderPosition()) / float(_slider_min_sample_distance->maximum());
  float absolute = _interval_min_sample_distance.minimum() + relative * _interval_min_sample_distance.length();
  
  QString as_str = QString::number(absolute, 'g', 4);
  _edit_min_sample_distance->setText(as_str);

  _settings.isosearch_min_sample_distance = absolute;
  apply_settings_to_renderer();
} 


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_max_sample_distance ()
{
  float relative = float(_slider_max_sample_distance->sliderPosition()) / float(_slider_max_sample_distance->maximum());
  float absolute = _interval_max_sample_distance.minimum() + relative * _interval_max_sample_distance.length();
  
  QString as_str = QString::number(absolute, 'g', 4);
  _edit_max_sample_distance->setText(as_str);

  _settings.isosearch_max_sample_distance = absolute;
  apply_settings_to_renderer();
} 


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_adaptive_sample_scale ()
{
  float relative = float(_slider_adaptive_sample_scale->sliderPosition()) / float(_slider_adaptive_sample_scale->maximum());
  float absolute = _interval_adaptive_sample_scale.minimum() + relative * _interval_adaptive_sample_scale.length();
  
  QString as_str = QString::number(absolute, 'g', 4);
  _edit_adaptive_sample_scale->setText(as_str);

  _settings.isosearch_sample_scale = absolute;
  apply_settings_to_renderer();
} 


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_max_binary_searches ()
{
  float relative = float(_slider_max_binary_searches->sliderPosition()) / float(_slider_max_binary_searches->maximum());
  unsigned absolute = _interval_max_binary_searches.minimum() + relative * _interval_max_binary_searches.length();
  
  QString as_str = QString::number(absolute);
  _edit_max_binary_searches->setText(as_str);

  _settings.isosearch_max_binary_steps = absolute;
  apply_settings_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_epsilon_newton_iteration()
{
  float relative = float(_slider_epsilon_newton_iteration->sliderPosition()) / float(_slider_epsilon_newton_iteration->maximum());
  float absolute = _interval_epsilon_newton_iteration.minimum() + relative * _interval_epsilon_newton_iteration.length();
  
  QString as_str = QString::number(absolute, 'g', 4);
  _edit_epsilon_newton_iteration->setText(as_str);

  _settings.newton_epsilon = absolute;
  apply_settings_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_max_newton_iteration ()
{
  float relative = float(_slider_max_newton_iteration->sliderPosition()) / float(_slider_max_newton_iteration->maximum());
  unsigned absolute = _interval_max_newton_iteration.minimum() + relative * _interval_max_newton_iteration.length();
  
  QString as_str = QString::number(absolute);
  _edit_max_newton_iteration->setText(as_str);

  _settings.newton_iterations = absolute;
  apply_settings_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_relative_isovalue ()
{
  float relative = float(_slider_relative_isovalue->sliderPosition()) / float(_slider_relative_isovalue->maximum());
  float absolute = _interval_relative_isovalue.minimum() + relative * _interval_relative_isovalue.length();
  
  QString as_str = QString::number(absolute, 'g', 4);
  _edit_relative_isovalue->setText(as_str);

  _settings.isosearch_relative_value = absolute;
  apply_settings_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_surface_transparency ()
{
  float relative = float(_slider_surface_transparency->sliderPosition()) / float(_slider_surface_transparency->maximum());
  float absolute = _interval_transparency.minimum() + relative * _interval_transparency.length();
  
  QString as_str = QString::number(absolute, 'g', 4);
  _edit_surface_transparency->setText(as_str);

  _settings.transparency_surface = absolute;
  apply_settings_to_renderer();
}



///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_isosurface_transparency ()
{
  float relative = float(_slider_isosurface_transparency->sliderPosition()) / float(_slider_isosurface_transparency->maximum());
  float absolute = _interval_transparency.minimum() + relative * _interval_transparency.length();
  
  QString as_str = QString::number(absolute, 'g', 4);
  _edit_isosurface_transparency->setText(as_str);

  _settings.transparency_isosurface = absolute;
  apply_settings_to_renderer();
}



///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_screenspace_newton_error()
{
  _settings.newton_screenspace_epsilon = _checkbox_screenspace_newton_error->checkState() == Qt::Checked;
  apply_settings_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_rendermode ( QString const& mode )
{  
  // convert nurbs model to bezier model
  apply_volume_to_renderer();

  // apply interface settings to renderer
  apply_interface_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::change_current_attribute ( QString const& mode )
{  
  // convert nurbs model to bezier model
  apply_volume_to_renderer();

  // apply interface settings to renderer
  apply_interface_to_renderer();
}


///////////////////////////////////////////////////////////////////////////////
void mainwindow::apply_settings_to_renderer()
{
  _glwindow->apply(_settings);
}

///////////////////////////////////////////////////////////////////////////////
void mainwindow::apply_default_settings ()
{
  _settings.cullface                      = true;
  _settings.fxaa                          = false;
  _settings.farplane                      = 10000.0f;
  _settings.nearplane                     = 1.0f;
  _settings.isosearch_adaptive_sampling   = true;
  _settings.isosearch_max_binary_steps    = 3;
  _settings.isosearch_max_sample_distance = 0.3f;
  _settings.isosearch_min_sample_distance = 0.002f;
  _settings.isosearch_relative_value      = 1.0f;
  _settings.isosearch_sample_scale        = 0.9f;
  _settings.newton_epsilon                = 0.0003f;
  _settings.newton_iterations             = 4;
  _settings.newton_screenspace_epsilon    = false;
  _settings.octree_max_depth              = 8;
  _settings.octree_max_volumes_per_node   = 128;
  _settings.transparency_isosurface       = 0.4f;
  _settings.transparency_surface          = 0.2f;

  _settings.visualization_props.show_samples_isosurface_intersection = false;
  _settings.visualization_props.show_samples_isosurface_intersection = false;
  _settings.visualization_props.show_face_intersections              = false;
  _settings.visualization_props.show_face_intersection_tests         = false;
  _settings.visualization_props.show_isosides                        = false;

  _settings.detect_implicit_extremum       = false;
  _settings.detect_implicit_inflection     = false;
  _settings.sample_based_face_intersection = true;
}


///////////////////////////////////////////////////////////////////////////////
void mainwindow::apply_settings_to_interface ()
{
  // apply slider values
  _slider_isosurface_transparency->setSliderPosition  ( int (_slider_width * _settings.transparency_isosurface       / _interval_transparency.length()));
  _slider_surface_transparency->setSliderPosition     ( int (_slider_width * _settings.transparency_surface          / _interval_transparency.length()));
  _slider_min_sample_distance->setSliderPosition      ( int (_slider_width * _settings.isosearch_min_sample_distance / _interval_min_sample_distance.length()));
  _slider_max_sample_distance->setSliderPosition      ( int (_slider_width * _settings.isosearch_max_sample_distance / _interval_max_sample_distance.length()));
  _slider_adaptive_sample_scale->setSliderPosition    ( int (_slider_width * _settings.isosearch_sample_scale        / _interval_adaptive_sample_scale.length()));
  _slider_max_binary_searches->setSliderPosition      ( int (_slider_width * _settings.isosearch_max_binary_steps    / _interval_max_binary_searches.length()));
  _slider_epsilon_newton_iteration->setSliderPosition ( int (_slider_width * _settings.newton_epsilon                / _interval_epsilon_newton_iteration.length()));
  _slider_max_newton_iteration->setSliderPosition     ( int (_slider_width * _settings.newton_iterations             / _interval_max_newton_iteration.length()));
  _slider_relative_isovalue->setSliderPosition        ( int (_slider_width * _settings.isosearch_relative_value      / _interval_relative_isovalue.length()));
  
  // apply bool values
  _checkbox_adaptive_sampling->setChecked         ( _settings.isosearch_adaptive_sampling );
  _checkbox_screenspace_newton_error->setChecked  ( _settings.newton_screenspace_epsilon );
  _checkbox_backface_culling->setChecked          ( _settings.cullface );
  _checkbox_show_isosides->setChecked             ( _settings.visualization_props.show_isosides );
  _checkbox_fxaa->setChecked                      ( _settings.fxaa );

  bool vsync_enabled;
  gpucast::gl::get_vsync(vsync_enabled);
  _checkbox_vsync->setChecked                     ( vsync_enabled );
}


///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::apply_interface_to_renderer ()
{
  change_adaptive_sampling            ();
  change_fxaa                         ();
  change_backface_culling             ();
  change_min_sample_distance          ();
  change_max_sample_distance          ();
  change_adaptive_sample_scale        ();
  change_max_binary_searches          ();
  change_epsilon_newton_iteration     ();
  change_max_newton_iteration         ();
  change_relative_isovalue            ();
  change_screenspace_newton_error     ();
  change_isosurface_transparency      ();
  change_surface_transparency         ();
                                      
  change_show_isosides                ();
  change_show_face_samples            ();
  change_show_face_intersections      ();
  change_show_face_intersection_tests ();

  change_newton_inflection            ();
  change_newton_extremum              ();
  change_sample_based_face_intersection ();
}

///////////////////////////////////////////////////////////////////////////////
void 
mainwindow::apply_attributes_to_interface()
{
  _box_choose_attribute->clear();

  BOOST_FOREACH ( auto attrib, _nurbsobject->attribute_list() )
  {
    _box_choose_attribute->addItem(attrib.c_str());
  }
}


///////////////////////////////////////////////////////////////////////////////
void mainwindow::stop_record()
{
  std::string file = boost::filesystem::current_path().string() + "/" + _lineedit_sequence_file->text().toStdString();
  _glwindow->stop_record(file);
}


///////////////////////////////////////////////////////////////////////////////
void mainwindow::run_sequence()
{
  std::string sequence_file = boost::filesystem::current_path().string() + "/" + _lineedit_sequence_file->text().toStdString();
  std::string result_file = boost::filesystem::current_path().string() + "/" + _lineedit_test_file->text().toStdString();
  _glwindow->run_sequence(sequence_file, result_file);
}


///////////////////////////////////////////////////////////////////////////////
glwidget::rendermode_t  
mainwindow::_get_mode ( QString const& mode ) const
{
  // stupid, but clash with QT with QString as key
  for (std::map<glwidget::rendermode_t, QString>::const_iterator i = _modemap.begin(); i != _modemap.end(); ++i) 
  {
    if ( i->second == mode ) {
      return i->first;
    }
  }

  return glwidget::face_interval_raycasting;
}



///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_actions()
{
  _action_exit                            = new QAction(tr("Exit"), this);
  _action_loadfile                        = new QAction(tr("Open Volume"), this);
  _action_loadsurffile                    = new QAction(tr("Open Surface"), this);
  _action_start_testing                   = new QAction(tr("Testrun"), this);

  // menu
  connect(_action_exit,                       SIGNAL( triggered()), this,         SLOT( close_window() ));
  connect(_action_loadfile,                   SIGNAL( triggered()), this,         SLOT( open_volume_file() ));
  connect(_action_loadsurffile,               SIGNAL( triggered()), this,         SLOT( open_surface_file() ));
  //connect(_action_start_testing,              SIGNAL( triggered()), _glwindow,    SLOT( run_test() ));
                                              
  // buttons                                  
  connect(_button_recompile,                      SIGNAL( released() ), _glwindow,    SLOT( recompile() ));
                                         
  // sliders                                      
  connect(_slider_epsilon_newton_iteration,       SIGNAL( valueChanged(int) ), this,  SLOT( change_epsilon_newton_iteration () ));
  connect(_slider_max_newton_iteration,           SIGNAL( valueChanged(int) ), this,  SLOT( change_max_newton_iteration() ));
  connect(_slider_max_binary_searches,            SIGNAL( valueChanged(int) ), this,  SLOT( change_max_binary_searches() ));
  connect(_slider_relative_isovalue,              SIGNAL( valueChanged(int) ), this,  SLOT( change_relative_isovalue() ));
  connect(_slider_min_sample_distance,            SIGNAL( valueChanged(int) ), this,  SLOT( change_min_sample_distance() ));
  connect(_slider_max_sample_distance,            SIGNAL( valueChanged(int) ), this,  SLOT( change_max_sample_distance() ));
  connect(_slider_adaptive_sample_scale,          SIGNAL( valueChanged(int) ), this,  SLOT( change_adaptive_sample_scale() ));
  connect(_slider_surface_transparency,           SIGNAL( valueChanged(int) ), this,  SLOT( change_surface_transparency() ));
  connect(_slider_isosurface_transparency,        SIGNAL( valueChanged(int) ), this,  SLOT( change_isosurface_transparency() ));

  connect(_button_start_sequence,                 SIGNAL( released() ), _glwindow,  SLOT( start_record() ));
  connect(_button_stop_sequence,                  SIGNAL( released() ), this,       SLOT( stop_record () ));
  connect(_button_run_sequence,                   SIGNAL( released() ), this,       SLOT( run_sequence() ));
  connect(_button_abort_sequence,                 SIGNAL( released() ), _glwindow,  SLOT( abort_sequence() ));
  connect(_button_apply,                          SIGNAL( released() ), this,       SLOT( apply_volume_to_renderer() ));
                                                  
  // box                                          
  connect(_box_choose_attribute,                  SIGNAL( currentIndexChanged(QString const&)), this,  SLOT( change_current_attribute (QString const&) ) );
  connect(_box_choose_rendermode,                 SIGNAL( currentIndexChanged(QString const&)), this,  SLOT( change_rendermode        (QString const&) ) );

  // checkbox                                     
  connect(_checkbox_adaptive_sampling,            SIGNAL( stateChanged(int)), this, SLOT( change_adaptive_sampling() ) );
  connect(_checkbox_show_isosides,                SIGNAL( stateChanged(int)), this, SLOT( change_show_isosides() ) );
  connect(_checkbox_fxaa,                         SIGNAL( stateChanged(int)), this, SLOT( change_fxaa() ) );
  connect(_checkbox_vsync,                        SIGNAL( stateChanged(int)), this, SLOT( change_vsync() ) );
  connect(_checkbox_backface_culling,             SIGNAL( stateChanged(int)), this, SLOT( change_backface_culling() ) );
  connect(_checkbox_screenspace_newton_error,     SIGNAL( stateChanged(int)), this, SLOT( change_screenspace_newton_error() ) );

  connect(_checkbox_show_face_samples,            SIGNAL( stateChanged(int)), this, SLOT( change_show_face_samples() ) );
  connect(_checkbox_show_face_intersections,      SIGNAL( stateChanged(int)), this, SLOT( change_show_face_intersections() ) );
  connect(_checkbox_show_face_intersection_tests, SIGNAL( stateChanged(int)), this, SLOT( change_show_face_intersection_tests() ) );
  connect(_checkbox_newton_inflection,            SIGNAL( stateChanged(int)), this, SLOT( change_newton_inflection() ) );
  connect(_checkbox_newton_extremum,              SIGNAL( stateChanged(int)), this, SLOT( change_newton_extremum() ) );
  connect(_checkbox_sample_based_face_intersection, SIGNAL( stateChanged(int)), this, SLOT( change_sample_based_face_intersection() ) );
}
      

///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_widgets( int argc, char** argv )
{
  /////////////////////////////////////
  // create QGLWidget as central element
  /////////////////////////////////////
  QGLFormat context_options;

  context_options.setVersion ( 4, 1 );
  //context_options.setProfile ( QGLFormat::CompatibilityProfile );
  context_options.setProfile ( QGLFormat::CoreProfile );
  context_options.setRgba    ( true );

  _glwindow = new glwidget   ( argc, argv, context_options, this );
  _glwindow->setFixedHeight(_height);
  _glwindow->setFixedWidth(_width);

  setCentralWidget(_glwindow);

  /////////////////////////////////////
  // create menus on right side of GLWindow
  /////////////////////////////////////
  int const widget_vertical_offset    = 10;
  int const widget_horizontal_offset  = 10;

  int const slider_height             = 14;

  int const labelvalue_width          = 50; 
  int const labeldescription_width    = 200;
  int const list_height               = 200;

  int const button_width = 290;
  int const button_height = 25;

  _menu = new QDockWidget(this);
  this->addDockWidget(Qt::RightDockWidgetArea, _menu);

  _menu->setFixedHeight(1024);
  _menu->setFixedWidth(768);

  _checkbox_adaptive_sampling         = new QCheckBox(_menu);
  _checkbox_show_isosides             = new QCheckBox(_menu);
  _checkbox_fxaa                      = new QCheckBox(_menu);
  _checkbox_vsync                     = new QCheckBox(_menu);
  _checkbox_backface_culling          = new QCheckBox(_menu);
  _checkbox_screenspace_newton_error  = new QCheckBox(_menu);

  _checkbox_show_face_samples              = new QCheckBox(_menu);
  _checkbox_show_face_intersections        = new QCheckBox(_menu);
  _checkbox_show_face_intersection_tests   = new QCheckBox(_menu);
  _checkbox_newton_inflection              = new QCheckBox(_menu);
  _checkbox_newton_extremum                = new QCheckBox(_menu);
  _checkbox_sample_based_face_intersection = new QCheckBox(_menu);

  _lineedit_sequence_file             = new QLineEdit(_menu);
  _lineedit_sequence_file->setText      ("default_sequence.txt");
  _lineedit_test_file                 = new QLineEdit(_menu);
  _lineedit_test_file->setText          ("default_test.txt");

  std::list<QCheckBox*> checkboxlist;
  checkboxlist.push_back(_checkbox_adaptive_sampling);
  checkboxlist.push_back(_checkbox_show_isosides);
  checkboxlist.push_back(_checkbox_fxaa);
  checkboxlist.push_back(_checkbox_vsync);
  checkboxlist.push_back(_checkbox_backface_culling);
  checkboxlist.push_back(_checkbox_screenspace_newton_error);
  checkboxlist.push_back(_checkbox_show_face_samples);       
  checkboxlist.push_back(_checkbox_show_face_intersections); 
  checkboxlist.push_back(_checkbox_show_face_intersection_tests); 
  checkboxlist.push_back(_checkbox_newton_inflection);     
  checkboxlist.push_back(_checkbox_newton_extremum);        
  checkboxlist.push_back(_checkbox_sample_based_face_intersection);        
       
  _slider_min_sample_distance       = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_max_sample_distance       = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_adaptive_sample_scale     = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_max_binary_searches       = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_epsilon_newton_iteration  = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_max_newton_iteration      = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_relative_isovalue         = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_surface_transparency      = new QSlider(Qt::Orientation::Horizontal, _menu);
  _slider_isosurface_transparency   = new QSlider(Qt::Orientation::Horizontal, _menu);

  std::list<QSlider*> sliderlist;
  sliderlist.push_back(_slider_min_sample_distance);
  sliderlist.push_back(_slider_max_sample_distance);
  sliderlist.push_back(_slider_adaptive_sample_scale);
  sliderlist.push_back(_slider_max_binary_searches);
  sliderlist.push_back(_slider_epsilon_newton_iteration);
  sliderlist.push_back(_slider_max_newton_iteration);
  sliderlist.push_back(_slider_relative_isovalue);
  sliderlist.push_back(_slider_surface_transparency);
  sliderlist.push_back(_slider_isosurface_transparency);
 
  _label_fps                          = new QLabel("ms/frame: fps:",            _menu);
  _label_adaptive_sampling            = new QLabel("Adaptive Sampling",         _menu);
  _label_show_isosides                = new QLabel("Classify by Iso Value",     _menu);
  _label_fxaa                         = new QLabel("FXAA",                      _menu);
  _label_vsync                        = new QLabel("VSync",                     _menu);
  _label_backface_culling             = new QLabel("Backface Culling",          _menu);
  _label_screenspace_newton_error     = new QLabel("Screenspace newton error",  _menu);
  _label_min_sample_distance          = new QLabel("Min. Sample Distance",      _menu);
  _label_max_sample_distance          = new QLabel("Max. Sample Distance",      _menu);
  _label_adaptive_sample_scale        = new QLabel("Sample Step Scale",         _menu);
  _label_max_binary_searches          = new QLabel("Max. Binary Search",        _menu);
  _label_epsilon_newton_iteration     = new QLabel("Epsilon Newton",            _menu);
  _label_max_newton_iteration         = new QLabel("Max. Newton Iterations",    _menu);
  _label_relative_isovalue            = new QLabel("Relative Isovalue",         _menu);
  _label_surface_transparency         = new QLabel("Surface Transparency",      _menu);
  _label_isosurface_transparency      = new QLabel("Isosurface Transparency",   _menu);
  _label_show_face_samples            = new QLabel("Show Face Detection Samples", _menu);
  _label_show_face_intersections      = new QLabel("Show Number of Face Intersections",        _menu);
  _label_show_face_intersection_tests = new QLabel("Show Number of Face Intersection Tests", _menu);

  _label_sample_based_face_intersection = new QLabel("Sample-based face detection", _menu);
  _label_newton_inflection            = new QLabel("Detect implicit inflection",      _menu);
  _label_newton_extremum              = new QLabel("Detect implicit extrema", _menu);
  

  std::list<QLabel*> labellist;
  labellist.push_back(_label_fps                      );
  labellist.push_back(_label_adaptive_sampling        );
  labellist.push_back(_label_show_isosides            );
  labellist.push_back(_label_fxaa                     );
  labellist.push_back(_label_vsync                    );
  labellist.push_back(_label_backface_culling         );
  labellist.push_back(_label_min_sample_distance      );
  labellist.push_back(_label_max_sample_distance      );
  labellist.push_back(_label_adaptive_sample_scale    );
  labellist.push_back(_label_max_binary_searches      );
  labellist.push_back(_label_epsilon_newton_iteration );
  labellist.push_back(_label_max_newton_iteration     );
  labellist.push_back(_label_relative_isovalue        );
  labellist.push_back(_label_surface_transparency     );   
  labellist.push_back(_label_isosurface_transparency  );   
  labellist.push_back(_label_screenspace_newton_error );   
  labellist.push_back(_label_show_face_samples );   
  labellist.push_back(_label_show_face_intersections );   
  labellist.push_back(_label_show_face_intersection_tests );   
  labellist.push_back(_label_newton_inflection );   
  labellist.push_back(_label_newton_extremum );   
  labellist.push_back(_label_sample_based_face_intersection );   
  
  _edit_min_sample_distance         = new QLabel ("", _menu);
  _edit_max_sample_distance         = new QLabel ("", _menu);
  _edit_adaptive_sample_scale       = new QLabel ("", _menu);
  _edit_max_binary_searches         = new QLabel ("", _menu);
  _edit_epsilon_newton_iteration    = new QLabel ("", _menu);
  _edit_max_newton_iteration        = new QLabel ("", _menu);
  _edit_relative_isovalue           = new QLabel ("", _menu);
  _edit_surface_transparency        = new QLabel ("", _menu);
  _edit_isosurface_transparency     = new QLabel ("", _menu);

  std::list<QLabel*> editlist;
  editlist.push_back(_edit_min_sample_distance      );
  editlist.push_back(_edit_max_sample_distance      );
  editlist.push_back(_edit_adaptive_sample_scale    );
  editlist.push_back(_edit_max_binary_searches      );
  editlist.push_back(_edit_epsilon_newton_iteration );
  editlist.push_back(_edit_max_newton_iteration     );
  editlist.push_back(_edit_relative_isovalue        );
  editlist.push_back(_edit_surface_transparency     );
  editlist.push_back(_edit_isosurface_transparency  );

  _button_start_sequence            = new QPushButton ( "start record", _menu );
  _button_stop_sequence             = new QPushButton ( "stop record",  _menu );
  _button_run_sequence              = new QPushButton ( "run test",     _menu );
  _button_abort_sequence            = new QPushButton ( "abort test",   _menu );
  _button_apply                     = new QPushButton ( "apply",        _menu );
  _button_recompile                 = new QPushButton ( "recompile",    _menu );


  std::for_each(sliderlist.begin(),   sliderlist.end(),   boost::bind(&QSlider::setMaximum,       _1, _slider_width));
  std::for_each(sliderlist.begin(),   sliderlist.end(),   boost::bind(&QSlider::setFixedHeight,   _1, slider_height));
  std::for_each(sliderlist.begin(),   sliderlist.end(),   boost::bind(&QSlider::setFixedWidth,    _1, _slider_width));
                                                                                                  
  std::for_each(editlist.begin(),     editlist.end(),     boost::bind(&QLabel::setAlignment,      _1, Qt::AlignRight ));
  std::for_each(editlist.begin(),     editlist.end(),     boost::bind(&QLabel::setFixedWidth,     _1, labelvalue_width ));
                                                                                                  
  std::for_each(labellist.begin(),    labellist.end(),    boost::bind(&QLabel::setAlignment,      _1, Qt::Alignment(Qt::AlignLeft & Qt::AlignVCenter) ));
  std::for_each(labellist.begin(),    labellist.end(),    boost::bind(&QLabel::setFixedWidth,     _1, labeldescription_width ));
  std::for_each(labellist.begin(),    labellist.end(),    boost::bind(&QLabel::setAlignment,      _1, Qt::AlignRight ));

  std::for_each(checkboxlist.begin(), checkboxlist.end(), boost::bind(&QCheckBox::setFixedHeight, _1, slider_height ));
  std::for_each(checkboxlist.begin(), checkboxlist.end(), boost::bind(&QCheckBox::setFixedWidth,  _1, _slider_width ));

  int pos = 2;

  _box_choose_rendermode = new QComboBox             ( _menu );
  _box_choose_rendermode->move                       ( widget_horizontal_offset, pos * (slider_height + widget_vertical_offset) - slider_height/2 );
  _box_choose_rendermode->setFixedHeight             ( button_height );
  _box_choose_rendermode->setFixedWidth              ( int(1.5 * button_width) );

  _button_recompile->move                           ( widget_horizontal_offset + int(1.5 * button_width), pos++ * (slider_height + widget_vertical_offset) - slider_height/2 );
  _button_recompile->setFixedHeight                 ( button_height );
  _button_recompile->setFixedWidth                  ( button_width / 2);

  _box_choose_attribute = new QComboBox             ( _menu );
  _box_choose_attribute->move                       ( widget_horizontal_offset, pos++ * (slider_height + widget_vertical_offset) - slider_height/2 );
  _box_choose_attribute->setFixedHeight             ( button_height );
  _box_choose_attribute->setFixedWidth              ( 2 * button_width );

  _checkbox_adaptive_sampling->move                 ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_adaptive_sampling->move                    ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_screenspace_newton_error->move          ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_screenspace_newton_error->move             ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_show_isosides->move                     ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_show_isosides->move                        ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);
   
  _checkbox_fxaa->move                              ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_fxaa->move                                 ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_vsync->move                             ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_vsync->move                                ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_backface_culling->move                  ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_backface_culling->move                     ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_show_face_samples->move                 ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_show_face_samples->move                    ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_show_face_intersections->move           ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_show_face_intersections->move              ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_show_face_intersection_tests->move      ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_show_face_intersection_tests->move         ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_sample_based_face_intersection->move    ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_sample_based_face_intersection->move       ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_newton_inflection->move                 ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_newton_inflection->move                    ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _checkbox_newton_extremum->move                   ( widget_horizontal_offset,                                                                 pos   * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_newton_extremum->move                      ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _edit_min_sample_distance->move                   ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_min_sample_distance->move                 ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_min_sample_distance->move                  ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);
                                                                                                                                            
  _edit_max_sample_distance->move                   ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_max_sample_distance->move                 ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_max_sample_distance->move                  ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);
                                                                                                                                           
  _edit_adaptive_sample_scale->move                 ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_adaptive_sample_scale->move               ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_adaptive_sample_scale->move                ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);
                                                                                                                                     
  _edit_max_binary_searches->move                   ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_max_binary_searches->move                 ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_max_binary_searches->move                  ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);
                                                                                                                                       
  _edit_epsilon_newton_iteration->move              ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_epsilon_newton_iteration->move            ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_epsilon_newton_iteration->move             ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);
                                                                                                                                 
  _edit_max_newton_iteration->move                  ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_max_newton_iteration->move                ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_max_newton_iteration->move                 ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);
                                                                                                                                           
  _edit_relative_isovalue->move                     ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_relative_isovalue->move                   ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_relative_isovalue->move                    ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _edit_surface_transparency->move                  ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_surface_transparency->move                ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_surface_transparency->move                 ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _edit_isosurface_transparency->move               ( widget_horizontal_offset,                                                                 pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _slider_isosurface_transparency->move             ( 2*widget_horizontal_offset + labelvalue_width,                                            pos * (slider_height + widget_vertical_offset) - slider_height/2);
  _label_isosurface_transparency->move              ( 2*widget_horizontal_offset + widget_horizontal_offset + labelvalue_width +_slider_width,  pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _label_fps->move                                  ( 2*widget_horizontal_offset,  list_height + pos++ * (slider_height + widget_vertical_offset) - slider_height/2);

  _lineedit_sequence_file->move                     ( widget_horizontal_offset + button_width/2, list_height + pos* (slider_height + widget_vertical_offset) - slider_height/2 );
  _lineedit_sequence_file->setFixedHeight           ( button_height );
  _lineedit_sequence_file->setFixedWidth            ( button_width );

  _button_start_sequence->move                      ( widget_horizontal_offset, list_height + pos++ * (slider_height + widget_vertical_offset) - slider_height/2 );
  _button_start_sequence->setFixedHeight            ( button_height );
  _button_start_sequence->setFixedWidth             ( button_width / 2 );

  _button_stop_sequence->move                       ( widget_horizontal_offset, list_height + pos++ * (slider_height + widget_vertical_offset) - slider_height/2 );
  _button_stop_sequence->setFixedHeight             ( button_height );
  _button_stop_sequence->setFixedWidth              ( button_width / 2 );

  _lineedit_test_file->move                         ( widget_horizontal_offset + button_width/2, list_height + pos* (slider_height + widget_vertical_offset) - slider_height/2 );
  _lineedit_test_file->setFixedHeight               ( button_height );
  _lineedit_test_file->setFixedWidth                ( button_width );

  _button_run_sequence->move                        ( widget_horizontal_offset, list_height + pos++ * (slider_height + widget_vertical_offset) - slider_height/2 );
  _button_run_sequence->setFixedHeight              ( button_height );
  _button_run_sequence->setFixedWidth               ( button_width / 2 );

  _button_abort_sequence->move                      ( widget_horizontal_offset, list_height + pos++ * (slider_height + widget_vertical_offset) - slider_height/2 );
  _button_abort_sequence->setFixedHeight            ( button_height );
  _button_abort_sequence->setFixedWidth             ( button_width / 2 );  

  _button_apply->move                      ( widget_horizontal_offset, list_height + pos++ * (slider_height + widget_vertical_offset) - slider_height/2 );
  _button_apply->setFixedHeight            ( button_height );
  _button_apply->setFixedWidth             ( button_width / 2 ); 
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_menus()
{
  _file_menu = menuBar()->addMenu(tr("File"));
  
  _file_menu->addSeparator();
  _file_menu->addAction   (_action_loadfile);
  _file_menu->addAction   (_action_loadsurffile);
  _file_menu->addAction   (_action_start_testing);
  _file_menu->addAction   (_action_exit);
}


///////////////////////////////////////////////////////////////////////////////
/* private */ void 
mainwindow::_create_statusbar()
{}



