/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : glwidget.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "glwidget.hpp"

#pragma warning(disable: 4127) // Qt conditional expression is constant

#include <mainwindow.hpp>

// system includes
#include <QtGui/QMouseEvent>
#include <QtOpenGL/QGLFormat>

#include <CL/cl.hpp>

#include <sstream>
#include <iostream>
#include <cctype>
#include <typeinfo>

#include <glpp/fragmentshader.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/util/timer.hpp>
#include <glpp/util/vsync.hpp>
#include <glpp/util/init_glew.hpp>
#include <glpp/error.hpp>
#include <glpp/util/contextinfo.hpp>

#include <glpp/test/dynamic_rotation.hpp>
#include <glpp/test/dynamic_scaling.hpp> 
#include <glpp/test/dynamic_translation.hpp>

#include <tml/parametric/point.hpp>
#include <tml/parametric/beziercurve.hpp>
#include <tml/parametric/beziersurface.hpp>
#include <tml/parametric/beziervolume.hpp>
#include <tml/parametric/nurbsvolume.hpp>

#include <gpucast/beziervolumeobject.hpp>
#include <gpucast/nurbsvolumeobject.hpp>
#include <gpucast/surface_renderer_gl.hpp>

#include <gpucast/isosurface/fragment/isosurface_renderer_unified_sampling.hpp>
#include <gpucast/isosurface/fragment/isosurface_renderer_interval_sampling.hpp>
#include <gpucast/isosurface/octree/isosurface_renderer_octreebased.hpp>
#include <gpucast/isosurface/grid/isosurface_renderer_gridbased.hpp>
#include <gpucast/isosurface/splat/isosurface_renderer_splatbased.hpp>

#include <gpucast/import/xml_loader.hpp>
#include <gpucast/import/xml2.hpp>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>


///////////////////////////////////////////////////////////////////////
glwidget::glwidget( int argc, char** argv, QGLFormat const& context_format, QWidget *parent)
 :  QGLWidget                       ( context_format, parent),
    _argc                           ( argc ),
    _argv                           ( argv ),
    _initialized                    ( false ),
    _fxaa                           ( false ),
    _surface_renderer               (),
    _isosurface_renderer            (),
    _width                          ( 1 ),
    _height                         ( 1 ),
    _test_sequence                  ( ),
    _test_sequence_drawtimes        (),
    _run_sequence                   ( false ),
    _record_sequence                ( false ),
    _test_sequence_filename         (),
    _test_results_filename          ()
{
  setFocusPolicy(Qt::StrongFocus);
}


///////////////////////////////////////////////////////////////////////
glwidget::~glwidget()
{}


///////////////////////////////////////////////////////////////////////
void                
glwidget::recompile ( )
{
  _init_shader();

  if ( _surface_renderer )    _surface_renderer->recompile();
  if ( _isosurface_renderer ) _isosurface_renderer->recompile();
}


///////////////////////////////////////////////////////////////////////
glwidget::surface_renderer_ptr const&
glwidget::surface_renderer () const
{
  return _surface_renderer;
}


///////////////////////////////////////////////////////////////////////
glwidget::isosurface_renderer_ptr const& 
glwidget::isosurface_renderer  () const
{
  return _isosurface_renderer;
}


///////////////////////////////////////////////////////////////////////
void  
glwidget::reset_trackball ()
{
  _trackball->reset();
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::boundingbox ( glwidget::boundingbox_t const& b )
{
  _boundingbox = b;
  _coordinate_system->set( glpp::vec4f(_boundingbox.min[0], _boundingbox.min[1], _boundingbox.min[2], 1.0f), float(0.1f * _boundingbox.size().abs()));
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::apply ( render_settings const& settings )
{
  if ( _isosurface_renderer )
  {
    _isosurface_renderer->adaptive_sample_scale      ( settings.isosearch_sample_scale );
    _isosurface_renderer->adaptive_sampling          ( settings.isosearch_adaptive_sampling );
    _isosurface_renderer->backface_culling           ( settings.cullface );
    _isosurface_renderer->isosurface_opacity         ( settings.transparency_isosurface );
    _isosurface_renderer->max_binary_searches        ( settings.isosearch_max_binary_steps );
    _isosurface_renderer->max_sample_distance        ( settings.isosearch_max_sample_distance );
    _isosurface_renderer->min_sample_distance        ( settings.isosearch_min_sample_distance );
    _isosurface_renderer->newton_epsilon             ( settings.newton_epsilon );
    _isosurface_renderer->newton_iterations          ( settings.newton_iterations );
    _isosurface_renderer->relative_isovalue          ( settings.isosearch_relative_value );
    _isosurface_renderer->screenspace_newton_error   ( settings.newton_screenspace_epsilon );
    _isosurface_renderer->visualization_props        ( settings.visualization_props );
    _isosurface_renderer->surface_opacity            ( settings.transparency_surface );

    _isosurface_renderer->detect_faces_by_sampling   ( settings.sample_based_face_intersection );
    _isosurface_renderer->detect_implicit_inflection ( settings.detect_implicit_inflection );
    _isosurface_renderer->detect_implicit_extremum   ( settings.detect_implicit_extremum );

    _isosurface_renderer->nearplane                  ( settings.nearplane );
    _isosurface_renderer->farplane                   ( settings.farplane );
                                                     
    _surface_renderer->nearplane                     ( settings.nearplane );
    _surface_renderer->farplane                      ( settings.farplane );
                                                     
    _fxaa = settings.fxaa;
  }
}



///////////////////////////////////////////////////////////////////////
void                    
glwidget::apply ( render_settings const& settings,
                  boost::shared_ptr<gpucast::nurbsvolumeobject> const& nurbsobject,
                  boost::shared_ptr<gpucast::beziervolumeobject> const& bezierobject,
                  std::string const& attribute,
                  std::string const& filename,
                  rendermode_t mode )
{
  // change renderer
  _change_isosurface_renderer ( mode );

  // apply settings to renderer
  apply ( settings );
  
  _boundingbox = nurbsobject->bbox();

  // try to load binary data
  std::string binary_extension;
  switch ( mode )
  {
    case splatting                : binary_extension = ".spb"; break;
    case unified_sampling         : binary_extension = ".usb"; break;
    case face_interval_raycasting : binary_extension = ".fib"; break;
    case octree_isosurface        : binary_extension = ".ocb"; break;
    case grid_isosurface          : binary_extension = ".gdb"; break;
    default : break;
  };

  boost::filesystem::path file(filename);
  boost::filesystem::path binary_name = (file.branch_path() / boost::filesystem::basename(file)).string() + attribute + binary_extension;
  std::string binary_name_str = binary_name.string();
  std::replace(binary_name_str.begin(), binary_name_str.end(), '/', '\\');

  if ( boost::filesystem::exists ( binary_name ) )
  {
    _isosurface_renderer->set(bezierobject, attribute);
    std::fstream ifstr ( binary_name_str.c_str(), std::ios_base::in | std::ios_base::binary );
    _isosurface_renderer->read(ifstr);
    ifstr.close();
  } else {
    _isosurface_renderer->init(bezierobject, attribute);
    std::fstream ofstr;
    ofstr.open ( binary_name_str.c_str(), std::ios_base::out | std::ios_base::binary );
    if ( ofstr.good() )
    {
      _isosurface_renderer->write(ofstr);
      ofstr.close();
    } else {
      std::cerr << "File open failed.\n";
      ofstr.close();
    }
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::start_record ()
{
  _test_sequence.clear();
  _record_sequence = true;
}


///////////////////////////////////////////////////////////////////////
void                   
glwidget::stop_record ( std::string const& file )
{
  _test_sequence_filename = file;
  _test_sequence.write(file);
  _record_sequence = false;
  _test_sequence.clear();
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::run_sequence ( std::string const& sequence_file, std::string const& file )
{
  _test_sequence.read(sequence_file);
  _test_results_filename = file;
  _test_sequence_drawtimes.clear();
  _run_sequence = true;
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::abort_sequence()
{
  _run_sequence = false;
}


///////////////////////////////////////////////////////////////////////
void glwidget::initializeGL()
{}


///////////////////////////////////////////////////////////////////////
void 
glwidget::resizeGL(int width, int height)
{
  _width  = width;
  _height = height;

  glViewport(0, 0, _width, _height);

  if ( !_initialized ) return;

  if ( _isosurface_renderer )     _isosurface_renderer->resize(_width, _height);
  if ( _surface_renderer )        _surface_renderer->resize(_width, _height);

  if ( _fxaa_input_color )
  {
    _fxaa_input_color->teximage(0, GL_RGBA32F, GLsizei(_width), GLsizei(_height), 0, GL_RGBA, GL_FLOAT, 0);
    _fxaa_input_color->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _fxaa_input_color->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  if ( _fxaa_input_depth )
  {
    _fxaa_input_depth->teximage(0, GL_DEPTH_COMPONENT32F_NV, GLsizei(_width), GLsizei(_height), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    _fxaa_input_depth->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _fxaa_input_depth->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  if ( _surface_color ) {
    _surface_color->teximage(0, GL_RGBA32F, GLsizei(_width), GLsizei(_height), 0, GL_RGBA, GL_FLOAT, 0);
    _surface_color->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _surface_color->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  if ( _surface_depth ) {
    _surface_depth->teximage(0, GL_DEPTH_COMPONENT32F_NV, GLsizei(_width), GLsizei(_height), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    _surface_depth->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _surface_depth->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  if ( _color_depth_texture )
  {
    _color_depth_texture->teximage(0, GL_RGBA32F, GLsizei(_width), GLsizei(_height), 0, GL_RGBA, GL_FLOAT, 0);
    _color_depth_texture->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _color_depth_texture->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  if ( _surface_fbo )
  {
    _surface_fbo->attach_texture ( *_surface_color, GL_COLOR_ATTACHMENT0_EXT );
    _surface_fbo->attach_texture ( *_surface_depth, GL_DEPTH_ATTACHMENT_EXT );

    _surface_fbo->bind();
    _surface_fbo->status();
    _surface_fbo->unbind();
  }
  
  if ( _color_depth_fbo )
  {
    _color_depth_fbo->attach_texture ( *_color_depth_texture, GL_COLOR_ATTACHMENT0_EXT );

    _color_depth_fbo->bind();
    _color_depth_fbo->status();
    _color_depth_fbo->unbind();
  }
  if ( _fxaa_input_fbo )
  {
    _fxaa_input_fbo->attach_texture ( *_fxaa_input_color, GL_COLOR_ATTACHMENT0_EXT );
    _fxaa_input_fbo->attach_texture ( *_fxaa_input_depth, GL_DEPTH_ATTACHMENT_EXT );

    _fxaa_input_fbo->bind();
    _fxaa_input_fbo->status();
    _fxaa_input_fbo->unbind();
  }

  if ( _isosurface_renderer )        _isosurface_renderer->set_external_texture       ( _color_depth_texture );
}

///////////////////////////////////////////////////////////////////////
void 
glwidget::paintGL()
{
  /////////////////////////////////////////
  // initialize renderer
  /////////////////////////////////////////
  if ( !_initialized )
  {
    _init();
    resizeGL(_width, _height);
  }
  
  /////////////////////////////////////////
  // 0. GL Setup - clear everything
  /////////////////////////////////////////
  glClearDepth(1.0);
  glDepthFunc(GL_LESS);

  /////////////////////////////////////////
  // 1. draw surface into FBO
  /////////////////////////////////////////
  glEnable ( GL_DEPTH_TEST );

  _surface_fbo->bind();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  glpp::vec3f translation = _boundingbox.center();
  glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, float(_boundingbox.size().abs()), 
                                     0.0f, 0.0f, 0.0f, 
                                     0.0f, 1.0f, 0.0f);

  glpp::matrix4f cam  = glpp::make_translation( _trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *_trackball->rotation() * 
                                                 glpp::make_translation(-translation[0], -translation[1], -translation[2]);

  glpp::matrix4f proj = glpp::frustum ( -float(_width)/_height, float(_width)/_height, -1.0f, 1.0f, _surface_renderer->nearplane(), _surface_renderer->farplane());
  glpp::matrix4f mv   = view * cam;
  glpp::matrix4f mvp  = mv * proj;

  /////////////////////////////////////////
  // performance tests 
  /////////////////////////////////////////
  if ( _run_sequence )
  {
    if ( _test_sequence.empty() ) 
    {
      // end test 
      _run_sequence = false; 
      std::ofstream fstr ( _test_results_filename, std::ios::out );
      if ( fstr.good() )
      {
        fstr << _test_sequence_filename << std::endl;
        std::for_each(_test_sequence_drawtimes.begin(), _test_sequence_drawtimes.end(), [&] ( float f ) { fstr << f << std::endl; } );
      } else {
        std::cerr << "Cannot open " << _test_sequence_filename << std::endl;
      }
    } else {
      // load matrices
      mv = _test_sequence.next().modelview;
      mvp = _test_sequence.next().modelviewprojection;
      _test_sequence.pop();
    }
  }
  
  _surface_renderer->modelviewmatrix    (mv);
  _surface_renderer->projectionmatrix   (proj);

  _surface_renderer->draw();

  _surface_fbo->unbind();

  /////////////////////////////////////////
  // 2. CUDA workaround - copy depth-component depth texture to floating point texture
  /////////////////////////////////////////
  _color_depth_fbo->bind();
  glClear(GL_COLOR_BUFFER_BIT );
  {
    _depth_copy_program->begin();
    {
      _depth_copy_program->set_uniform1i ( "width", _width );
      _depth_copy_program->set_uniform1i ( "height", _height );
      _depth_copy_program->set_texture2d ( "color_texture", *_surface_color, 0 );
      _depth_copy_program->set_texture2d ( "depth_texture", *_surface_depth, 1 );
      _quad->draw();
    }
    _depth_copy_program->end();
  }
  _color_depth_fbo->unbind();

  /////////////////////////////////////////
  // 3. draw volume into fxaa_fbo
  /////////////////////////////////////////
  _fxaa_input_fbo->bind();
  {
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    _isosurface_renderer->modelviewmatrix  (mv);
    _isosurface_renderer->projectionmatrix (proj);
    _isosurface_renderer->draw             ();  

    // draw coordinate system
    _base_program->begin();
    _base_program->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _coordinate_system->draw();
    _base_program->end();
  }
  _fxaa_input_fbo->unbind();

  /////////////////////////////////////////
  // 3. draw volume 
  /////////////////////////////////////////

  // render into drawbuffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  _fbo_program->begin();
  _fbo_program->set_texture2d( "colorbuffer", *_fxaa_input_color, 0);
  //_fbo_program->set_texture2d( "colorbuffer", *_color_depth_texture, 0);
  _fbo_program->set_uniform1i( "width",                          GLint(_width));
  _fbo_program->set_uniform1i( "height",                         GLint(_height));
  _fbo_program->set_uniform1i( "enable", GLint(_fxaa) );
  _quad->draw();
  _fbo_program->end();

  /////////////////////////////////////////
  // tell mainwindow - a frame has been rendered
  /////////////////////////////////////////
  mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
  if ( mainwin != 0 ) 
  {
    double t = mainwin->frame();

    if ( _run_sequence )
    {
      _test_sequence_drawtimes.push_back(t);
    }
  }

  // redraw
  this->update();
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::mousePressEvent(QMouseEvent *event)
{
  enum glpp::eventhandler::button b;

  switch (event->button()) {
    case Qt::MouseButton::LeftButton    : b = glpp::eventhandler::left; break;
    case Qt::MouseButton::RightButton   : b = glpp::eventhandler::right; break;
    case Qt::MouseButton::MiddleButton  : b = glpp::eventhandler::middle; break;
    default : return;
  }

  _trackball->mouse(b, glpp::eventhandler::press, event->x(), event->y());
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::mouseReleaseEvent(QMouseEvent *event)
{
  enum glpp::eventhandler::button b;

  switch (event->button()) {
    case Qt::MouseButton::LeftButton    : b = glpp::eventhandler::left; break;
    case Qt::MouseButton::RightButton   : b = glpp::eventhandler::right; break;
    case Qt::MouseButton::MiddleButton  : b = glpp::eventhandler::middle; break;
    default : return;
  }

  _trackball->mouse(b, glpp::trackball::release, event->x(), event->y());
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::mouseMoveEvent(QMouseEvent *event)
{
  _trackball->motion(event->x(), event->y());

  glpp::vec3f translation = _boundingbox.center();
  glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, float(_boundingbox.size().abs()), 
                                     0.0f, 0.0f, 0.0f, 
                                     0.0f, 1.0f, 0.0f);

  glpp::matrix4f cam  = glpp::make_translation( _trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *_trackball->rotation() * 
                                                 glpp::make_translation(-translation[0], -translation[1], -translation[2]);

  glpp::matrix4f proj = glpp::frustum ( -float(_width)/_height, float(_width)/_height, -1.0f, 1.0f, _isosurface_renderer->nearplane(), _isosurface_renderer->farplane());
  glpp::matrix4f mv   = view * cam;
  glpp::matrix4f mvp  = mv * proj;

  if ( _record_sequence )
  {
    _test_sequence.add ( mv, glpp::inverse(mv), mvp, glpp::inverse(mvp), mv.normalmatrix() );
  }
}



///////////////////////////////////////////////////////////////////////
/* virtual */ void  
glwidget::keyPressEvent ( QKeyEvent* /*event*/)
{}


///////////////////////////////////////////////////////////////////////
/* virtual */ void  
glwidget::keyReleaseEvent ( QKeyEvent* event )
{
  char key = event->key();

  if (event->modifiers() != Qt::ShiftModifier) {
    key = std::tolower(key);
  }

  switch ( key ) 
  {
    case 'r' :
      recompile();
      break;
    case 'P' :
      {
        boost::shared_ptr<gpucast::fragmentlist_generator> fraglist_gen = boost::static_pointer_cast<gpucast::fragmentlist_generator>(_isosurface_renderer);
        if ( fraglist_gen )
        {
          int current_pagesize = fraglist_gen->pagesize();
          fraglist_gen->pagesize( current_pagesize + 1 );
          std::cout << "current pagesize : " << fraglist_gen->pagesize() << std::endl;
        }
        break;
      }
    case 'p' :
     {
       boost::shared_ptr<gpucast::fragmentlist_generator> fraglist_gen = boost::static_pointer_cast<gpucast::fragmentlist_generator>(_isosurface_renderer);
       if ( fraglist_gen )
       {
         int current_pagesize = fraglist_gen->pagesize();
         fraglist_gen->pagesize( std::max(1, current_pagesize - 1)); 
         std::cout << "current pagesize : " << fraglist_gen->pagesize() << std::endl;
       }
       break;
     }
    case 'w' :
     {
       boost::shared_ptr<gpucast::fragmentlist_generator> fraglist_gen = boost::static_pointer_cast<gpucast::fragmentlist_generator>(_isosurface_renderer);
       if ( fraglist_gen )
       {
          int current_grid_width = fraglist_gen->allocation_grid_width();
          fraglist_gen->allocation_grid_width( std::max(1, current_grid_width - 1)); 
          std::cout << "current allocation_grid_width : " << fraglist_gen->allocation_grid_width() << std::endl;
       }
       break;
     }

    case 'W' :
     {
       boost::shared_ptr<gpucast::fragmentlist_generator> fraglist_gen = boost::static_pointer_cast<gpucast::fragmentlist_generator>(_isosurface_renderer);
       if ( fraglist_gen )
       {
         int current_grid_width = fraglist_gen->allocation_grid_width();
         fraglist_gen->allocation_grid_width( current_grid_width + 1 ); 
         std::cout << "current allocation_grid_width : " << fraglist_gen->allocation_grid_width() << std::endl;
       }
       break;
     }

    case 'h' :
     {
       boost::shared_ptr<gpucast::fragmentlist_generator> fraglist_gen = boost::static_pointer_cast<gpucast::fragmentlist_generator>(_isosurface_renderer);
       if ( fraglist_gen )
       {
         int current_grid_height = fraglist_gen->allocation_grid_height();
         fraglist_gen->allocation_grid_height( std::max(1, current_grid_height - 1)); 
         std::cout << "current allocation_grid_height : " << fraglist_gen->allocation_grid_height() << std::endl;
       }
       break;
     }

    case 'H' :
     {
       boost::shared_ptr<gpucast::fragmentlist_generator> fraglist_gen = boost::static_pointer_cast<gpucast::fragmentlist_generator>(_isosurface_renderer);
       if ( fraglist_gen )
       {
         int current_grid_height = fraglist_gen->allocation_grid_height();
         fraglist_gen->allocation_grid_height( current_grid_height + 1 ); 
         std::cout << "current allocation_grid_height : " << fraglist_gen->allocation_grid_height() << std::endl;
       }
       break;
     }

     default : 
       break;// do nothing
      
  }
}

//////////////////////////////////////////////////////////////////////
void
glwidget::_change_isosurface_renderer ( rendermode_t mode )
{
  _isosurface_renderer.reset();

  switch ( mode ) 
  {
    case splatting                : _isosurface_renderer.reset ( new gpucast::isosurface_renderer_splatbased(_argc, _argv)); break;
    case octree_isosurface        : _isosurface_renderer.reset ( new gpucast::isosurface_renderer_octreebased(_argc, _argv)); break;
    case face_interval_raycasting : _isosurface_renderer.reset ( new gpucast::isosurface_renderer_interval_sampling(_argc, _argv)); break;
    case unified_sampling         : _isosurface_renderer.reset ( new gpucast::isosurface_renderer_unified_sampling(_argc, _argv)); break;
    case grid_isosurface          : _isosurface_renderer.reset ( new gpucast::isosurface_renderer_gridbased(_argc, _argv)); break;
    default                       : _isosurface_renderer.reset ( new gpucast::isosurface_renderer_unified_sampling(_argc, _argv)); break;
  };
  
  if ( _color_depth_texture->width() != _width || _color_depth_texture->height() != _height )
  {
    resizeGL(_width, _height);
  }

  _isosurface_renderer->resize(_width, _height);
  _isosurface_renderer->set_external_texture(_color_depth_texture);
}

///////////////////////////////////////////////////////////////////////
void
glwidget::_init()
{
  if ( _initialized ) 
  {
    return;
  } else {
    _initialized = true;
  }

  mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
  if ( mainwin )
  {
    mainwin->show();
    mainwin->frame();
  } else {
    _initialized = false;
    return;
  }

  glpp::init_glew ( std::cout );

  glpp::print_contextinfo               ( std::cout );

  glpp::print_extensions                ( std::cout );
  
  // initialize renderer
  _surface_renderer.reset               ( new gpucast::surface_renderer_gl(_argc, _argv));

  // init trackball and coordinatesystem + shader
  _trackball.reset            ( new glpp::trackball(0.3f, 0.2f, 0.1f));
  _coordinate_system.reset    ( new glpp::coordinate_system(0, 1) );
  _base_program.reset         ( new glpp::program );

  // initialize shaders
  _init_shader                ();

  // init framebufferobject for fxaa
  _fxaa_input_fbo.reset      ( new glpp::framebufferobject );
  _fxaa_input_color.reset    ( new glpp::texture2d );
  _fxaa_input_depth.reset    ( new glpp::texture2d );

  _color_depth_fbo.reset     ( new glpp::framebufferobject );
  _color_depth_texture.reset ( new glpp::texture2d );

  _surface_fbo.reset         ( new glpp::framebufferobject );
  _surface_color.reset       ( new glpp::texture2d );
  _surface_depth.reset       ( new glpp::texture2d );
  
  _quad.reset                ( new glpp::plane(0, -1, 1) );

  // create default renderer
  _change_isosurface_renderer ( glwidget::face_interval_raycasting );
}

///////////////////////////////////////////////////////////////////////
void
glwidget::_init_shader()
{
  gpucast::isosurface_renderer_interval_sampling tmp(_argc, _argv);
  tmp.init_program ( _base_program,       "/base/base.vert", "/base/base.frag" );
  tmp.init_program ( _fbo_program,        "/base/render_from_texture.vert", "/base/render_from_texture.frag" );
  tmp.init_program ( _depth_copy_program, "/base/render_from_texture.vert", "/base/copy_depth.frag" );
}
