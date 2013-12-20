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
#include <glpp/util/init_glew.hpp>
#include <glpp/util/timer.hpp>
#include <glpp/util/vsync.hpp>
#include <glpp/error.hpp>

#include <tml/parametric/point.hpp>
#include <tml/parametric/beziercurve.hpp>
#include <tml/parametric/beziersurface.hpp>
#include <tml/parametric/beziervolume.hpp>
#include <tml/parametric/nurbsvolume.hpp>

#include <gpucast/beziersurfaceobject.hpp>
#include <gpucast/nurbssurfaceobject.hpp>
#include <gpucast/surface_converter.hpp>
#include <gpucast/import/xml_loader.hpp>
#include <gpucast/import/igs_loader.hpp>
#include <gpucast/import/mesh3d_loader.hpp>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>


#define FXAA_ANTIALIASING

///////////////////////////////////////////////////////////////////////
glwidget::glwidget( int argc, char** argv, QGLFormat const& context_format, QWidget *parent)
 :  QGLWidget         ( context_format, parent),
    _argc             ( argc ),
    _argv             ( argv ),
    _initialized      ( false ),
    _renderer         (),
    _frames           ( 0 ),
    _time             ( 0.0 ),
    _background       ( 0.2f, 0.2f, 0.2f ),
    _cullface         ( true ),
    _fxaa             ( false ),
    _ambient_occlusion( false ),
    _aoradius         ( 30.0f ),
    _aosamples        ( 500 ) 
{
  setFocusPolicy(Qt::StrongFocus);
}


///////////////////////////////////////////////////////////////////////
glwidget::~glwidget()
{}


///////////////////////////////////////////////////////////////////////
void                
glwidget::open ( std::list<std::string> const& files )
{  
  _trackball->reset();
  if ( files.empty() ) return;

  // clear old objects
  _objects.clear();
  _renderer->clear();

  // open file(s) and set new boundingbox
  std::for_each(files.begin(), files.end(), [&] ( std::string const& file ) {
                                                                              tml::axis_aligned_boundingbox<tml::point3d> bbox;
                                                                              _openfile(file, bbox);
                                                                              if ( file == files.front() ) 
                                                                              {
                                                                                _boundingbox = bbox; // reset bounding box
                                                                              } else {
                                                                                _boundingbox.merge(bbox); // extend bounding box
                                                                              }
                                                                            } );
  //_openfile ( file, _boundingbox );
}


///////////////////////////////////////////////////////////////////////
void                
glwidget::add ( std::list<std::string> const& files )
{ 
  if ( files.empty() ) return;

  std::for_each(files.begin(), files.end(), [&] ( std::string const& file ) 
                                                  {
                                                    // create temporary new bounding box of added objects
                                                    tml::axis_aligned_boundingbox<tml::point3d> bbox;
                                                    _openfile ( file, bbox );
                                                    _boundingbox.merge ( bbox );
                                                  } );
}



///////////////////////////////////////////////////////////////////////
void                
glwidget::recompile ( )
{
  _renderer->recompile();
  _renderer->init_program( _fbo_program,  "/base/render_from_texture_sao.vert", "/base/render_from_texture_sao.frag" );
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
  
  glViewport(0, 0, GLsizei(_width), GLsizei(_height));

  // if renderer already initialized -> resize
  if (_renderer) 
  {
    _renderer->resize(int(_width), int(_height));

    _colorattachment.reset(new glpp::texture2d);
    _colorattachment->teximage(0, GL_RGBA32F, GLsizei(_width), GLsizei(_height), 0, GL_RGBA, GL_FLOAT, 0);

    _depthattachment.reset(new glpp::texture2d);
    _depthattachment->teximage(0, GL_DEPTH32F_STENCIL8 , GLsizei(_width), GLsizei(_height), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    _fbo.reset(new glpp::framebufferobject);
    _fbo->attach_texture (*_colorattachment, GL_COLOR_ATTACHMENT0_EXT);
    _fbo->attach_texture (*_depthattachment, GL_DEPTH_STENCIL_ATTACHMENT);

    _fbo->bind();
    _fbo->status();
    _fbo->unbind();
    _generate_random_texture();
  }


}


///////////////////////////////////////////////////////////////////////
void 
glwidget::paintGL()
{
  // draw pre-evaluated stuff... 
  if (!_initialized)
  {
    glpp::init_glew();

    // init data, buffers and shader
    _init();
    _initialized = true;

    // try to update main widget
    mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
    if (mainwin) {
      mainwin->update_interface();
    } else {
      std::cerr << "glwidget::paintGL(): Could not cast to mainwindow widget" << std::endl;
    }
  }

  glpp::timer t;
  t.start();

  if (_cullface) 
  {
    glEnable(GL_CULL_FACE);
  } else {
    glDisable(GL_CULL_FACE);
  }

  _fbo->bind();

  glClearColor(_background[0], _background[1], _background[2], 1.0f);

  glClearDepth ( 1.0f );
  //glDepthFunc  ( GL_LESS );
  glEnable     ( GL_DEPTH_TEST );

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  
  _renderer->nearplane        ( 0.01f * _boundingbox.size().abs() );
  _renderer->farplane         ( 1.5f  * _boundingbox.size().abs() );

  glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, float(_boundingbox.size().abs()), 
                                     0.0f, 0.0f, 0.0f, 
                                     0.0f, 1.0f, 0.0f);

  glpp::vec3f translation = _boundingbox.center();

  glpp::matrix4f model    = glpp::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *_trackball->rotation() * 
                            glpp::make_translation(-translation[0], -translation[1], -translation[2]);

  glpp::matrix4f proj = glpp::perspective(60.0f, float(_width)/_height, _renderer->nearplane(), _renderer->farplane());
  glpp::matrix4f mv   = view * model;
  glpp::matrix4f mvp  = proj * mv;
  glpp::matrix4f nm   = mv.normalmatrix();

  _renderer->modelviewmatrix  (mv);
  _renderer->projectionmatrix (proj);
  std::for_each(_objects.begin(), _objects.end(), [&] ( std::pair<gpucast::surface_renderer_gl::drawable_ptr, std::string> const& p ) { _renderer->draw(p.first); } );
  //_renderer->draw             ();  

  glFinish(); 

  _fbo->unbind();

  ++_frames;

  // pass fps to window
  t.stop();
  glpp::time_duration elapsed = t.result();
  double drawtime_seconds = elapsed.fractional_seconds + elapsed.seconds; // discard minutes
  _time += drawtime_seconds;

  // show message and reset counter if more than 1s passed
  if ( _time > 0.5 || _frames > 20 ) 
  {
    mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
    if (mainwin) 
    {
      mainwin->show_fps ( double(_frames) / _time );
    }
    _frames = 0;
    _time   = 0.0;
  }

  // render into drawbuffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  
  _fbo_program->begin();
  
  _fbo_program->set_uniform_matrix4fv ( "modelviewprojectioninverse", 1, false, &_renderer->modelviewprojectioninverse()[0]);
  _fbo_program->set_uniform_matrix4fv ( "modelviewprojection", 1,        false, &_renderer->modelviewprojection()[0]);
  _fbo_program->set_texture2d         ( "colorbuffer",       *_colorattachment,   0);
  _fbo_program->set_texture2d         ( "depthbuffer",       *_depthattachment,   1);
  _fbo_program->set_texture2d         ( "random_texture",    *_aorandom_texture,  2);
  _fbo_program->set_texturebuffer     ( "ao_sample_offsets", *_aosample_offsets,  3);
  
  _sample_linear->bind(0);
  _sample_linear->bind(1);
  _sample_nearest->bind(2);
  
  _fbo_program->set_uniform1i         ( "ao_enable",         int(_ambient_occlusion) );
  _fbo_program->set_uniform1i         ( "ao_samples",        _aosamples );
  _fbo_program->set_uniform1f         ( "ao_radius",         _aoradius );
  _fbo_program->set_uniform1i         ( "fxaa",              int(_fxaa) );
  
  _fbo_program->set_uniform1i         ( "width",             GLsizei(_width));
  _fbo_program->set_uniform1i         ( "height",            GLsizei(_height));
  _quad->draw();
  
  _fbo_program->end();
  
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

  //mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());

  switch ( key ) 
  {
    case 'r' :
      recompile();
      break;
    case 'c' :
      _renderer->cube_mapping(!_renderer->cube_mapping());
      break;
    case 'I' :
      _renderer->newton_iterations(_renderer->newton_iterations() + 1);
      break;
    case 'i' :
      _renderer->newton_iterations(std::max(std::size_t(1), _renderer->newton_iterations() - 1));
      break;
    case 'd' :
      _renderer->diffuse_mapping(!_renderer->diffuse_mapping());
      break;
    case 'y' :
      {
        std::size_t surfsize = 0;
        std::size_t dbsize = 0;
        std::size_t cmbsize = 0;
        _renderer->memory_usage ( dbsize, cmbsize, surfsize );
        std::cout << "db size : " << dbsize << " Bytes " << std::endl;
        std::cout << "cmb size : " << cmbsize << " Bytes " << std::endl;
        std::cout << "percent : " << float(cmbsize)/dbsize << " Bytes " << std::endl;
        std::cout << "surface size : " << surfsize << " Bytes " << std::endl;
      break;
      }
    case 's' :
      _renderer->sphere_mapping(!_renderer->sphere_mapping());
      break;
    case 't' :
      _renderer->trimming(!_renderer->trimming());
      break;
    case 'n' :
      _renderer->raycasting(!_renderer->raycasting());
      break;
    case 'b' :
      _cullface = !_cullface;
      break;
    case 'a' :
      {
        switch ( _renderer->trim_approach() )
        {
          case gpucast::surface_renderer::double_binary_partition :
            std::cout << "switching trim mode to contourmap_binary" << std::endl;
            _renderer->trim_approach(gpucast::surface_renderer::contourmap_binary);
            break;
          case gpucast::surface_renderer::contourmap_binary       :
            std::cout << "switching trim mode to contourmap_kdtree" << std::endl;
            _renderer->trim_approach(gpucast::surface_renderer::contourmap_kdtree);
            break;
          case gpucast::surface_renderer::contourmap_kdtree       : 
            std::cout << "switching trim mode to double_binary_partition" << std::endl;
            _renderer->trim_approach(gpucast::surface_renderer::double_binary_partition);
            break;
          default :
          break;
        }
        break;
      }
    //case 'm':
    //  _CrtDumpMemoryLeaks();
    //  break;

    default : 
      break;// do nothing 
  }
}



///////////////////////////////////////////////////////////////////////
  void                    
  glwidget::load_spheremap                ( )
  {
    QString in_image_path = QFileDialog::getOpenFileName(this, tr("Open Image"), ".", tr("Image Files (*.jpg *.jpeg *.hdr *.bmp *.png *.tiff *.tif)"));
    _renderer->spheremap(in_image_path.toStdString());
  }


  ///////////////////////////////////////////////////////////////////////
  void                    
  glwidget::load_diffusemap               ( )
  {
    QString in_image_path = QFileDialog::getOpenFileName(this, tr("Open Image"), ".", tr("Image Files (*.jpg *.jpeg *.hdr *.bmp *.png *.tiff *.tif)"));
    _renderer->diffusemap(in_image_path.toStdString());
  }

  ///////////////////////////////////////////////////////////////////////
  void                    
  glwidget::spheremapping                 ( int i )
  {
    _renderer->sphere_mapping(i);
  }


  ///////////////////////////////////////////////////////////////////////
  void                    
  glwidget::diffusemapping                ( int i )
  {
    _renderer->diffuse_mapping(i);
  }


  ///////////////////////////////////////////////////////////////////////
  void                      
  glwidget::fxaa                          ( int i )
  {
    _fxaa = i;
  }


  
  ///////////////////////////////////////////////////////////////////////
  void                      
  glwidget::vsync                          ( int i )
  {
    glpp::set_vsync(i != 0);
  }



  ///////////////////////////////////////////////////////////////////////
  void                      
  glwidget::ambient_occlusion             ( int i )
  {
    _ambient_occlusion = i;
  }




///////////////////////////////////////////////////////////////////////
void
glwidget::_init()
{
  glpp::init_glew ();
  _print_contextinfo();

  _renderer.reset        ( new gpucast::surface_renderer_gl ( _argc, _argv ) );
  _renderer->resize      ( int(_width), int(_height));

  _trackball.reset       ( new glpp::trackball(0.6f, 0.3f, 0.1f) );

  _fbo.reset             ( new glpp::framebufferobject );
  _depthattachment.reset ( new glpp::texture2d );
  _colorattachment.reset ( new glpp::texture2d );
  _aorandom_texture.reset( new glpp::texture2d );
  _aosample_offsets.reset( new glpp::texturebuffer );
  _quad.reset            ( new glpp::plane(0, -1, 1) );

  _sample_linear.reset   ( new glpp::sampler );
  _sample_linear->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
  _sample_linear->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
  _sample_linear->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _sample_linear->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  _sample_nearest.reset  ( new glpp::sampler );
  _sample_nearest->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
  _sample_nearest->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
  _sample_nearest->parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  _sample_nearest->parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  _generate_random_texture();
  _generate_ao_sampletexture();

  _fbo_program.reset ( new glpp::program );

  recompile();
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::_generate_ao_sampletexture ()
{
  std::vector<float> offsets(4 * TOTAL_RANDOM_SAMPLES); // 8 x random_access, 4 elements per offset * samples
  glpp::time_duration t0;
  glpp::timer t;
  t.time(t0);
  std::srand(unsigned(t0.fractional_seconds*1000.0));
  
  std::generate(offsets.begin(), offsets.end(), [&] () { return 2.0f * (std::rand()/float(RAND_MAX) - 0.5f); } );
  _aosample_offsets->update(offsets.begin(), offsets.end());
  _aosample_offsets->format(GL_RGBA32F);
}


///////////////////////////////////////////////////////////////////////
void
glwidget::_generate_random_texture()
{
  std::vector<unsigned> random_values ( _width * _height );
  std::generate (random_values.begin(), random_values.end(), [&] () { return std::rand()%(TOTAL_RANDOM_SAMPLES - _aosamples); } );
  _aorandom_texture->teximage(0, GL_R32I, int(_width), int(_height), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &random_values[0]);
}


///////////////////////////////////////////////////////////////////////
void
glwidget::_print_contextinfo()
{
  char* gl_version   = (char*)glGetString(GL_VERSION);
  char* glsl_version = (char*)glGetString(GL_SHADING_LANGUAGE_VERSION);

  GLint context_profile;
  glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &context_profile);

  std::cout << "OpenGL Version String : " << gl_version << std::endl;
  std::cout << "GLSL Version String   : " << glsl_version << std::endl;

  switch (context_profile) {
    case GL_CONTEXT_CORE_PROFILE_BIT :
      std::cout << "Core Profile" << std::endl; break;
    case GL_CONTEXT_COMPATIBILITY_PROFILE_BIT :
      std::cout << "Compatibility Profile" << std::endl; break;
    default :
      std::cout << "Unknown Profile" << std::endl;
  };
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::_openfile ( std::string const& file, tml::axis_aligned_boundingbox<tml::point3d>& bbox )
{
  gpucast::igs_loader         igsloader;
  gpucast::surface_converter  nurbsconverter;
  std::string                 extension = boost::filesystem::extension(file);
    
  if ( extension == ".dat" )
  { 
    gpucast::mesh3d_loader                           mesh3d_loader;
    boost::shared_ptr<gpucast::nurbssurfaceobject>   nurbsobject ( new gpucast::nurbssurfaceobject );
    boost::shared_ptr<gpucast::beziersurfaceobject>  bezierobject = _renderer->create();

    mesh3d_loader.load(file, nurbsobject);
    nurbsconverter.convert(nurbsobject, bezierobject);

    glpp::material mat;
    mat.randomize(0.05f, 1.0f, 0.1f, 20.0f, 1.0f);

    bezierobject->material(mat);
    _objects.insert(std::make_pair(bezierobject, file));
    bbox = bezierobject->bbox();
  }

  if ( extension == ".igs" )
  { 
    boost::shared_ptr<gpucast::nurbssurfaceobject>   nurbsobject ( new gpucast::nurbssurfaceobject );
    boost::shared_ptr<gpucast::beziersurfaceobject>  bezierobject = _renderer->create();

    igsloader.load(file, nurbsobject);
    nurbsconverter.convert(nurbsobject, bezierobject);

    glpp::material mat;
    mat.randomize(0.05f, 1.0f, 0.1f, 20.0f, 1.0f);

    bezierobject->material(mat);
    _objects.insert(std::make_pair(bezierobject, file));
    bbox = bezierobject->bbox();
  }

  if ( extension == ".cfg" )
  {
    std::ifstream ifstr(file.c_str());
    typedef std::vector<std::pair<glpp::material, std::string> > file_map_t;
    file_map_t filemap;
    glpp::material current_material;

    if (ifstr.good()) 
    {
      std::string line;
      while (ifstr) 
      {
        std::getline(ifstr, line);

        if (!line.empty()) 
        {
          std::istringstream sstr(line);
          std::string qualifier;
          sstr >> qualifier;

          // if not comment line
          if (qualifier.size() > 0) 
          {
            if (qualifier.at(0) != '#') 
            {
              // define material
              if (qualifier == "material") 
              {
                _parse_material_conf(sstr, current_material);
              }
              // load igs file
              if (qualifier == "object") 
              {
                if (sstr) 
                {
                  std::string filename;
                  sstr >> filename;
                  filemap.push_back(std::make_pair(current_material, filename));
                }
              }
              if (qualifier == "background") 
              {
                if (sstr) {
                  _parse_background(sstr, _background);
                }
              }
            }
          }
        }
      }
    }

    for (file_map_t::iterator i = filemap.begin(); i != filemap.end(); ++i) 
    {
      boost::shared_ptr<gpucast::nurbssurfaceobject>   nurbsobject ( new gpucast::nurbssurfaceobject );
      boost::shared_ptr<gpucast::beziersurfaceobject>  bezierobject = _renderer->create();

      igsloader.load(i->second, nurbsobject);
      nurbsconverter.convert(nurbsobject, bezierobject);

      bezierobject->material(i->first);
      _objects.insert(std::make_pair(bezierobject, file));

      if ( i == filemap.begin() )
      { 
        bbox = bezierobject->bbox();
      } else {
        bbox.merge(bezierobject->bbox());
      }
    }

    ifstr.close();
  }
}


///////////////////////////////////////////////////////////////////////////////
void
glwidget::_parse_material_conf(std::istringstream& sstr, glpp::material& mat) const
{
  float ar, ag, ab, dr, dg , db, sr, sg, sb, shine, opac;

  // ambient coefficients
  _parse_float(sstr, ar);
  _parse_float(sstr, ag);
  _parse_float(sstr, ab);

  // diffuse coefficients
  _parse_float(sstr, dr);
  _parse_float(sstr, dg);
  _parse_float(sstr, db);

  // specular coefficients
  _parse_float(sstr, sr);
  _parse_float(sstr, sg);
  _parse_float(sstr, sb);

  // shininess
  _parse_float(sstr, shine);

  // opacity
  if (_parse_float(sstr, opac)) {
    mat.ambient   = glpp::vec3f(ar, ag, ab);
    mat.diffuse   = glpp::vec3f(dr, dg, db);
    mat.specular  = glpp::vec3f(sr, sg, sb);
    mat.shininess = shine;
    mat.opacity   = opac;
	} else {
	  std::cerr << "application::read_material(): material definition incomplete. discarding.\n usage: material ar ab ag   dr dg db  sr sg sb  shininess   opacity";
	}
}


///////////////////////////////////////////////////////////////////////////////
bool
glwidget::_parse_float(std::istringstream& sstr, float& result) const
{
  if (sstr) {
    sstr >> result;
    return true;
  } else {
    return false;
  }
}

///////////////////////////////////////////////////////////////////////////////
void
glwidget::_parse_background(std::istringstream& sstr, glpp::vec3f& bg) const
{
  float r, g, b;
  sstr >> r >> g >> b;
  bg = glpp::vec3f(r, g, b);
}
