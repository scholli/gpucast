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

#include <sstream>
#include <iostream>
#include <cctype>
#include <typeinfo>

#include <gpucast/core/config.hpp>

#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/util/init_glew.hpp>
#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/util/vsync.hpp>
#include <gpucast/gl/util/resource_factory.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/primitives/bezierobject_renderer.hpp>

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziercurve.hpp>
#include <gpucast/math/parametric/beziersurface.hpp>
#include <gpucast/math/parametric/beziervolume.hpp>
#include <gpucast/math/parametric/nurbsvolume.hpp>

#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/import/igs_loader.hpp>
#include <gpucast/core/import/igs_loader.hpp>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>


///////////////////////////////////////////////////////////////////////
glwidget::glwidget(int argc, char** argv, QGLFormat const& context_format, QWidget *parent)
  : QGLWidget(context_format, parent),
  _argc(argc),
  _argv(argv),
  _initialized(false),
  _frames(0),
  _background(0.2f, 0.2f, 0.2f),
  _fxaa(false),
  _ambient_occlusion(false),
  _aoradius(30.0f),
  _aosamples(500)
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

  // open file(s) and set new boundingbox
  std::for_each(files.begin(), files.end(), [&] ( std::string const& file ) {
                                                                              gpucast::math::axis_aligned_boundingbox<gpucast::math::point3d> bbox;
                                                                              _openfile(file, bbox);
                                                                              std::cout << "Opening " << file << " : Bbox = " << bbox << std::endl;

                                                                              if ( file == files.front() ) 
                                                                              {
                                                                                _boundingbox = bbox; // reset bounding box
                                                                              } else {
                                                                                _boundingbox.merge(bbox); // extend bounding box
                                                                              }
                                                                            } );
  _update_memory_usage();
}


///////////////////////////////////////////////////////////////////////
void                
glwidget::add ( std::list<std::string> const& files )
{ 
  if ( files.empty() ) return;

  std::for_each(files.begin(), files.end(), [&] ( std::string const& file ) 
                                                  {
                                                    // create temporary new bounding box of added objects
                                                    gpucast::math::axis_aligned_boundingbox<gpucast::math::point3d> bbox;
                                                    _openfile ( file, bbox );
                                                    _boundingbox.merge ( bbox );
                                                  } );
  _update_memory_usage();
}

///////////////////////////////////////////////////////////////////////
void glwidget::remove(std::list<std::string> const& files)
{
  if (files.empty()) return;

  std::for_each(files.begin(), files.end(), [&](std::string const& file)
  {
    BOOST_LOG_TRIVIAL(info) << "removing : " << file << std::endl;
    _objects.erase(file);
  }
  );
}

///////////////////////////////////////////////////////////////////////
std::map<unsigned, double> glwidget::surfaces_by_degree() const
{
  std::map<unsigned, double> result;

  for (auto const& o : _objects) {
    for (auto const& surface_pair : o.second->object().order_surfaces()) {
      result[surface_pair.first] += surface_pair.second;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////
std::map<unsigned, double> glwidget::curves_by_degree() const
{
  std::map<unsigned, double> result;

  for (auto const& o : _objects) {
    for (auto const& curve_pair : o.second->object().order_trimcurves()) {
      result[curve_pair.first] += curve_pair.second;
    }
  }

  return result;
}


///////////////////////////////////////////////////////////////////////
void                
glwidget::recompile ( )
{
  _renderer->recompile();

  gpucast::gl::resource_factory program_factory;

  _fxaa_program = program_factory.create_program({
    {gpucast::gl::vertex_stage,   "resources/glsl/base/render_from_texture.vert"},
    {gpucast::gl::fragment_stage, "resources/glsl/base/render_from_texture.frag"}
  });

  _ssao_program = program_factory.create_program({
    { gpucast::gl::vertex_stage,   "resources/glsl/base/render_from_texture.vert" },
    { gpucast::gl::fragment_stage, "resources/glsl/base/render_from_texture_sao.frag" }
  });
}


///////////////////////////////////////////////////////////////////////
void glwidget::initializeGL()
{
  glEnable(GL_DEPTH_TEST);
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::resizeGL(int width, int height)
{
  _width  = width;
  _height = height;

  std::cout << "reisze : " << width << " : " << height << std::endl;

  if (!_initialized) {
    return;
  }
  
  glViewport(0, 0, GLsizei(_width), GLsizei(_height));

  auto renderer = _renderer;
  renderer->set_resolution(_width, _height);

  // create texture for FBO
  if (!_colorattachment || !_depthattachment) {
    _colorattachment.reset(new gpucast::gl::texture2d);
    _depthattachment.reset(new gpucast::gl::texture2d);
  }
    
  _colorattachment->teximage(0, GL_RGBA32F, GLsizei(_width), GLsizei(_height), 0, GL_RGBA, GL_FLOAT, 0);
  _depthattachment->teximage(0, GL_DEPTH32F_STENCIL8, GLsizei(_width), GLsizei(_height), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

  renderer->attach_custom_textures(_colorattachment, _depthattachment);

  _generate_random_texture();

  // update SSAO texture
  if (!_ssao_attachment) {
    _ssao_attachment.reset(new gpucast::gl::texture2d);
  }
  _ssao_attachment->teximage(0, GL_RGB32F, GLsizei(_width), GLsizei(_height), 0, GL_RGB, GL_FLOAT, 0);

  if (!_ssao_depth) {
    _ssao_depth.reset(new gpucast::gl::renderbuffer);
  }
  _ssao_depth->set(GL_DEPTH32F_STENCIL8, GLsizei(_width), GLsizei(_height));

  // create FBO
  if (!_ssao_fbo) {
    _ssao_fbo.reset(new gpucast::gl::framebufferobject);
  }
  
  _ssao_fbo->attach_renderbuffer(*_ssao_depth, GL_DEPTH_STENCIL_ATTACHMENT);
  _ssao_fbo->attach_texture(*_ssao_attachment, GL_COLOR_ATTACHMENT0_EXT);

  _ssao_fbo->bind();
  _ssao_fbo->status();
  _ssao_fbo->unbind();
}




///////////////////////////////////////////////////////////////////////
void 
glwidget::paintGL()
{
  //////////////////////////////////////////////
  // lazy ressource initialization
  //////////////////////////////////////////////
  if (!_initialized)
  {
    // init data, buffers and shader
    _init();
    _initialized = true;

    // try to update main widget
    mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
    if (mainwin) {
      mainwin->update_interface();
      mainwin->set_defaults();
    } else {
      std::cerr << "glwidget::paintGL(): Could not cast to mainwindow widget" << std::endl;
    }
  }

  //////////////////////////////////////////////
  // measure timing
  //////////////////////////////////////////////
  if (!_gputimer) _gputimer = std::make_shared<gpucast::gl::timer_query>();
  if (!_gputimer_postprocess) _gputimer_postprocess = std::make_shared<gpucast::gl::timer_query>();
  if (!_cputimer) _cputimer = std::make_shared<gpucast::gl::timer>();

  gpucast::gl::timer total_frame_timer;
  total_frame_timer.start();
  _cputimer->start();
  bool query_this_frame = _gputimer->result_fetched();

  if (query_this_frame) {
    _gputimer->begin();
  }

  //////////////////////////////////////////////
  // bind FBO
  //////////////////////////////////////////////
  if (!_colorattachment || !_depthattachment) {
    resizeGL(_width, _height);
  }

  //////////////////////////////////////////////
  // compute render parameters and view setup
  //////////////////////////////////////////////
  float nearplane = 0.01f * _boundingbox.size().abs();
  float farplane  = 5.0f  * _boundingbox.size().abs();

  gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, float(_boundingbox.size().abs()), 
                                     0.0f, 0.0f, 0.0f, 
                                     0.0f, 1.0f, 0.0f);

  gpucast::math::vec3f translation = _boundingbox.center();

  gpucast::math::matrix4f model    = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *_trackball->rotation() * 
                                     gpucast::math::make_translation(-translation[0], -translation[1], -translation[2]);

  gpucast::math::matrix4f proj = gpucast::math::perspective(50.0f, float(_width) / _height, nearplane, farplane); 

  //////////////////////////////////////////////
  // GL state setup
  //////////////////////////////////////////////
  auto renderer = _renderer;
  glEnable(GL_DEPTH_TEST);
  glClearColor(_background[0], _background[1], _background[2], 1.0f);

  //////////////////////////////////////////////
  // apply camera setup to renderer
  //////////////////////////////////////////////
  renderer->set_nearfar(nearplane, farplane);
  renderer->set_resolution(_width, _height);

  renderer->current_modelmatrix(model);
  renderer->current_viewmatrix(view);
  renderer->current_projectionmatrix(proj);

  _cputimer->stop();
  if (_frames%100==0) std::cout << "Initialization time : " << _cputimer->result().as_seconds() * 1000.0 << std::endl;

  _cputimer->start();
  renderer->begin_draw();
  glFinish();
  _cputimer->stop();
  if (_frames % 100 == 0) std::cout << "Predraw time : " << _cputimer->result().as_seconds() * 1000.0 << std::endl;

  _cputimer->start();
  for (auto const& o : _objects) {
    o.second->draw(*_renderer);
  }
  glFinish();
  _cputimer->stop();
  if (_frames % 100 == 0) std::cout << "Object draw time : " << _cputimer->result().as_seconds() * 1000.0 << std::endl;

  _cputimer->start();
  renderer->end_draw();
  glFinish();
  _cputimer->stop();
  if (_frames % 100 == 0) std::cout << "postdraw time : " << _cputimer->result().as_seconds() * 1000.0 << std::endl;

  _cputimer->start();
  //////////////////////////////////////////////
  // readback debug info if enabled
  //////////////////////////////////////////////
  if (_renderer->enable_counting()) {
    auto debug_info = _renderer->get_debug_count();

    auto estimate = _renderer->get_fragment_estimate();
    auto estimate_total = std::accumulate(estimate.begin(), estimate.end(), 0);
    mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
    if (mainwin) {
      mainwin->update_count(debug_info.triangles, debug_info.fragments, debug_info.culled_triangles, debug_info.trimmed_fragments, estimate_total);
    }
  }
  glFinish();
  _cputimer->stop();
  if (_frames % 100 == 0) std::cout << "readback time : " << _cputimer->result().as_seconds() * 1000.0 << std::endl;
 

  //////////////////////////////////////////////
  // pass fps to GUI window
  //////////////////////////////////////////////
  _cputimer->start();

  if (query_this_frame) {
    _gputimer->end();
  }

  if (_gputimer->is_available()) {
    ++_frames;
    _gputime += _gputimer->time_in_ms(false);

  }
#if 1


  gpucast::math::matrix4f mv = view * model;
  gpucast::math::matrix4f mvp = proj * mv;
  gpucast::math::matrix4f mvpi = gpucast::math::inverse(mvp);

  query_this_frame = _gputimer_postprocess->result_fetched();
  if (query_this_frame) {
    ++_frames_postprocess;
    _gputimer_postprocess->begin();
  }

  //////////////////////////////////////////////
  // SSAO
  //////////////////////////////////////////////

  if (_ambient_occlusion) {
    _ssao_fbo->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    _ssao_program->begin();
    { 
      // setting uniforms
      _ssao_program->set_texture2d("depthbuffer", *_depthattachment, 5);
      _sample_linear->bind(5);

      _ssao_program->set_texture2d("randomtexture", *_aorandom_texture, 6);
      _sample_linear->bind(6);

      _ssao_program->set_uniform_matrix4fv("modelviewprojectioninverse", 1, false, &mvpi[0]);
      _fxaa_program->set_uniform_matrix4fv("modelview", 1, false, &mv[0]);

      _ssao_program->set_uniform1i("width", GLsizei(_width));
      _ssao_program->set_uniform1i("height", GLsizei(_height));

      _ssao_program->set_uniform1f("nearclip", nearplane);
      _ssao_program->set_uniform1f("farclip", farplane);

      // draw
      _quad->draw();
    }
    _ssao_program->end();

    _ssao_fbo->unbind();
  }
  //////////////////////////////////////////////
  // render into drawbuffer
  //////////////////////////////////////////////
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glClearColor(_background[0], _background[1], _background[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  _fxaa_program->begin();

  _fxaa_program->set_uniform_matrix4fv("modelviewprojectioninverse", 1, false, &mvpi[0]);
  _fxaa_program->set_uniform_matrix4fv("modelviewprojection", 1, false, &mvp[0]);

  _fxaa_program->set_texture2d("colorbuffer", *_colorattachment, 1);
  _fxaa_program->set_texture2d("depthbuffer", *_depthattachment, 2);
  _fxaa_program->set_texture2d("ssao_texture", *_ssao_attachment, 3);

  _sample_linear->bind(1);
  _sample_linear->bind(2);
  _sample_linear->bind(3);

  _fxaa_program->set_uniform1i("fxaa_mode", int(_fxaa));
  _fxaa_program->set_uniform1i("ssao_mode", int(_ambient_occlusion));
  _fxaa_program->set_uniform1i("width", GLsizei(_width));
  _fxaa_program->set_uniform1i("height", GLsizei(_height));
  _quad->draw();

  _fxaa_program->end();

  if (query_this_frame) {
    _gputimer_postprocess->end();
  }

#endif

  if (_gputimer_postprocess->is_available()) {
    _postprocess += _gputimer_postprocess->time_in_ms(false);
  }

  glFinish();
  _cputimer->stop();
  if (_frames % 100 == 0) std::cout << "Frame blit/FXAA time : " << _cputimer->result().as_seconds() * 1000.0 << std::endl;

  // show message and reset counter if more than 1s passed
  if (_gputime > 500 || _frames > 10)
  {
    mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
    if (mainwin)
    {
      mainwin->show_fps( (1000.0*_cputime) / _frames, _gputime / _frames, _postprocess / _frames_postprocess);
    }
    _frames = 0;

    _cputime = 0.0;
    _gputime = 0.0;
  }

  total_frame_timer.stop();
  if (_frames % 100 == 0) std::cout << "Total frame time : " << total_frame_timer.result().as_seconds() * 1000.0 << std::endl;

  total_frame_timer.start();
  this->update();

  total_frame_timer.stop();
  if (_frames % 100 == 0) std::cout << "GUI update : " << total_frame_timer.result().as_seconds() * 1000.0 << std::endl;
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::mousePressEvent(QMouseEvent *event)
{
  enum gpucast::gl::eventhandler::button b;

  switch (event->button()) {
    case Qt::MouseButton::LeftButton    : b = gpucast::gl::eventhandler::left; break;
    case Qt::MouseButton::RightButton   : b = gpucast::gl::eventhandler::right; break;
    case Qt::MouseButton::MiddleButton  : b = gpucast::gl::eventhandler::middle; break;
    default : return;
  }

  _trackball->mouse(b, gpucast::gl::eventhandler::press, event->x(), event->y());
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::mouseReleaseEvent(QMouseEvent *event)
{
  enum gpucast::gl::eventhandler::button b;

  switch (event->button()) {
    case Qt::MouseButton::LeftButton    : b = gpucast::gl::eventhandler::left; break;
    case Qt::MouseButton::RightButton   : b = gpucast::gl::eventhandler::right; break;
    case Qt::MouseButton::MiddleButton  : b = gpucast::gl::eventhandler::middle; break;
    default : return;
  }

  _trackball->mouse(b, gpucast::gl::trackball::release, event->x(), event->y());
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

  switch (key)
  {
    case 'r':
      _renderer->recompile();
      break;
  }


  for ( auto const& o : _objects) 
  {
    switch (key)
    {
      case 'q':
        switch (o.second->trimming()) {
        case gpucast::beziersurfaceobject::no_trimming:
          o.second->trimming(gpucast::beziersurfaceobject::curve_binary_partition);
          std::cout << "Switching trim approach to: curve_binary_partition" << std::endl;
          break;
        case gpucast::beziersurfaceobject::curve_binary_partition :
          o.second->trimming(gpucast::beziersurfaceobject::contour_binary_partition);
          std::cout << "Switching trim approach to: contour_binary_partition" << std::endl;
          break;
        case gpucast::beziersurfaceobject::contour_binary_partition:
          o.second->trimming(gpucast::beziersurfaceobject::contour_kd_partition);
          std::cout << "Switching trim approach to: contour_kd_partition" << std::endl;
          break;
        case gpucast::beziersurfaceobject::contour_kd_partition:
          o.second->trimming(gpucast::beziersurfaceobject::contour_list);
          std::cout << "Switching trim approach to: contour_list" << std::endl;
          break;
        case gpucast::beziersurfaceobject::contour_list:
          o.second->trimming(gpucast::beziersurfaceobject::no_trimming);
          std::cout << "Switching trim approach to: no_trimming" << std::endl;
          break;
        };
        break;
      case 'I':
        o.second->raycasting_max_iterations(o.second->raycasting_max_iterations() + 1);
        break;
      case 'i':
        o.second->raycasting_max_iterations(std::max(1U, o.second->raycasting_max_iterations() - 1));
        break;
      case 'n':
        o.second->enable_raycasting(!o.second->enable_raycasting());
        break;
      case 'b':
        o.second->culling(!o.second->culling());
        break;
      default:
        break;// do nothing 
    }
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
  void glwidget::spheremapping(int i)
  {
    _renderer->spheremapping(i);
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
    gpucast::gl::set_vsync(i != 0);
  }


  ///////////////////////////////////////////////////////////////////////
  void                      
  glwidget::ambient_occlusion             ( int i )
  {
    _ambient_occlusion = i;
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::apply_material(std::string const& name, gpucast::math::vec4f const& ambient, gpucast::math::vec4f const& diffuse, gpucast::math::vec4f const& specular, float shininess, float opacity)
  {
    auto o = _objects.at(name);
    if (o) {
      gpucast::gl::material new_mat{ ambient, diffuse, specular, shininess, opacity };
      o->set_material(new_mat);
      BOOST_LOG_TRIVIAL(info) << "Set Material : A[" << name << ambient << " ], D[" << diffuse << "], S[" << specular << "], shininess=" << shininess << " , opacity= " << opacity << std::endl;
    }
    else {
      BOOST_LOG_TRIVIAL(error) << "Could not find object " << name << std::endl;
    }
  }
  ///////////////////////////////////////////////////////////////////////
  void glwidget::spheremap(std::string const& filepath)
  {
    _renderer->spheremap(filepath);
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::diffusemap(std::string const& filepath)
  {
    _renderer->diffusemap(filepath);
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::enable_pretessellation(int i)
  {
    _renderer->enable_pretessellation(i);
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::conservative_rasterization(int h)
  {
    _renderer->enable_conservative_rasterization(h);
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::holefilling(int h)
  {
    _renderer->enable_holefilling(h);
  }


  ///////////////////////////////////////////////////////////////////////
  void glwidget::backface_culling(int i) {
    for (auto o : _objects) {
      o.second->culling(i != 0);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::rendermode(gpucast::gl::bezierobject::render_mode mode)
  {
    for (auto o : _objects) {
      o.second->rendermode(mode);
      o.second->fillmode(_fillmode);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::fillmode(gpucast::gl::bezierobject::fill_mode mode)
  {
    _fillmode = mode;
    for (auto o : _objects) {
      o.second->fillmode(mode);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void
  glwidget::antialiasing(gpucast::gl::bezierobject::anti_aliasing_mode i)
  {
    _antialiasing = i;
    _renderer->antialiasing(i);

    for (auto o : _objects) {
      o.second->trimming(_trimming);
    }

    resizeGL(_width, _height);
  }


  ///////////////////////////////////////////////////////////////////////
  void
  glwidget::trimming(gpucast::beziersurfaceobject::trim_approach_t t)
  {
    _trimming = t;
    for (auto o : _objects) {
      o.second->trimming(t);
    }
    if (_initialized) recompile();
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::preclassification(int i)
  {
    for (auto o : _objects) {
      o.second->init(0, 0, i);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void
  glwidget::enable_counter(int i) {
    _renderer->enable_counting(i);
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::trim_max_bisections(int i)
  {
    for (auto o : _objects) {
      o.second->trimming_max_bisections(i);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::trim_error_tolerance(float f)
  {
    for (auto o : _objects) {
      o.second->trimming_error_tolerance(f);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::tesselation_max_pixel_error(float f)
  {
    for (auto o : _objects) {
      o.second->tesselation_max_pixel_error(f);
    }
  }
  ///////////////////////////////////////////////////////////////////////
  void glwidget::tesselation_max_geometric_error(float f)
  {
    for (auto o : _objects) {
      o.second->tesselation_max_geometric_error(f);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::raycasting_max_iterations(int i)
  {
    for (auto o : _objects) {
      o.second->raycasting_max_iterations(i);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::raycasting_error_tolerance(float f)
  {
    for (auto o : _objects) {
      o.second->raycasting_error_tolerance(f);
    }
  }

  ///////////////////////////////////////////////////////////////////////
  void glwidget::enable_triangular_tesselation(int enable)
  {
    _renderer->enable_triangular_tesselation(enable);
  }

///////////////////////////////////////////////////////////////////////
void
glwidget::_init()
{
  gpucast::gl::init_glew ();
  _print_contextinfo();

  _renderer = std::make_shared<gpucast::gl::bezierobject_renderer>();

  _renderer->add_search_path("../../../");
  _renderer->add_search_path("../../");
  _renderer->recompile();

  _trackball.reset       ( new gpucast::gl::trackball );

  _aorandom_texture.reset( new gpucast::gl::texture2d );
  _aosample_offsets.reset( new gpucast::gl::texturebuffer );
  _quad.reset            ( new gpucast::gl::plane(0, -1, 1) );

  _sample_linear.reset   ( new gpucast::gl::sampler );
  _sample_linear->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
  _sample_linear->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
  _sample_linear->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _sample_linear->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  _sample_nearest.reset  ( new gpucast::gl::sampler );
  _sample_nearest->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
  _sample_nearest->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
  _sample_nearest->parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  _sample_nearest->parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  _generate_random_texture();
  _generate_ao_sampletexture();

  recompile();
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::_generate_ao_sampletexture ()
{
  const int size = 512;
  std::vector<float> offsets(2 * size * size); // 8 x random_access, 4 elements per offset * samples
  gpucast::gl::time_duration t0;
  gpucast::gl::timer t;
  t.time(t0);
  std::srand(unsigned(t0.fractional_seconds*1000.0));
  
  std::generate(offsets.begin(), offsets.end(), [&] () { return 2.0f * (std::rand()/float(RAND_MAX) - 0.5f); } );
  _aosample_offsets->update(offsets.begin(), offsets.end());
  _aosample_offsets->format(GL_RG32F);
}


///////////////////////////////////////////////////////////////////////
void
glwidget::_generate_random_texture()
{
  const int size = 512;
  std::vector<unsigned> random_values (size * size);
  std::generate (random_values.begin(), random_values.end(), [&] () { return std::rand()%(TOTAL_RANDOM_SAMPLES - _aosamples); } );
  _aorandom_texture->teximage(0, GL_R32I, int(size), int(size), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &random_values[0]);
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
glwidget::_openfile ( std::string const& file, gpucast::math::axis_aligned_boundingbox<gpucast::math::point3d>& bbox )
{
  gpucast::igs_loader         igsloader;

  gpucast::surface_converter  nurbsconverter;
  std::string                 extension = boost::filesystem::extension(file);

  if ( extension == ".igs" )
  { 
    auto nurbsobjects = igsloader.load(file);

    for (auto no : nurbsobjects) 
    {
      auto bezierobject = std::make_shared<gpucast::beziersurfaceobject>();
      nurbsconverter.convert(no, bezierobject);

      bezierobject->init();
      auto drawable = std::make_shared<gpucast::gl::bezierobject>(*bezierobject);

      gpucast::gl::material mat;

      auto is_black = [](gpucast::math::vec3f const& c) { return c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0; };

      // apply color if it has one, except black
      if (no->color() && !is_black(no->color().get())) {
        mat.diffuse = gpucast::math::vec4f{ no->color().get()[0], no->color().get()[1], no->color().get()[2], 1.0 };
        mat.ambient = gpucast::math::vec4f{ 0.0, 0.0, 0.0, 0.0 };
        mat.specular = gpucast::math::vec4f{ 0.0, 0.0, 0.0, 0.0 };
        mat.shininess = 20.0;
        mat.opacity = 1.0;
      }
      else {
        mat.randomize(0.05f, 1.0f, 0.1f, 20.0f, 1.0f);
      }
      std::cout << "Setting material...\n";
      mat.print(std::cout);

      drawable->set_material(mat);
      
      

      _objects.insert(std::make_pair(file, drawable));

      bbox = bezierobject->bbox();
    }
  }

  if ( extension == ".cfg" )
  {
    std::ifstream ifstr(file.c_str());
    typedef std::vector<std::pair<gpucast::gl::material, std::string> > file_map_t;
    file_map_t filemap;
    gpucast::gl::material current_material;

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
      auto nurbsobjects = igsloader.load(i->second);

      for (auto nurbsobject : nurbsobjects) 
      {
        auto bezierobject = std::make_shared<gpucast::beziersurfaceobject>();
        nurbsconverter.convert(nurbsobject, bezierobject);

        bezierobject->init();

        auto drawable = std::make_shared<gpucast::gl::bezierobject>(*bezierobject);
        drawable->set_material(i->first);
        bbox = bezierobject->bbox();

        _objects.insert(std::make_pair (file, drawable));

        if (i == filemap.begin())
        {
          bbox = bezierobject->bbox();
        }
        else {
          bbox.merge(bezierobject->bbox());
        }
      }
    }

    ifstr.close();
  }
}


///////////////////////////////////////////////////////////////////////////////
void
glwidget::_parse_material_conf(std::istringstream& sstr, gpucast::gl::material& mat) const
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
    mat.ambient   = gpucast::math::vec3f(ar, ag, ab);
    mat.diffuse   = gpucast::math::vec3f(dr, dg, db);
    mat.specular  = gpucast::math::vec3f(sr, sg, sb);
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
glwidget::_parse_background(std::istringstream& sstr, gpucast::math::vec3f& bg) const
{
  float r, g, b;
  sstr >> r >> g >> b;
  bg = gpucast::math::vec3f(r, g, b);
}

///////////////////////////////////////////////////////////////////////////////
void
glwidget::_update_memory_usage() 
{
  gpucast::beziersurfaceobject::memory_usage usage;

  for (auto const& i : _objects) {
    usage += i.second->object().get_memory_usage();
  }
  mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
  if (mainwin) {
    mainwin->show_memory_usage(usage);
  }
}