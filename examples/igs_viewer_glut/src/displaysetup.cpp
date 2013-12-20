/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : displaysetup.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// i/f header
#include "displaysetup.hpp"

// header, system
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

// header, project
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/error.hpp>

#include <application.hpp>



///////////////////////////////////////////////////////////////////////////////
displaysetup::displaysetup(std::string const& name,
                           unsigned width,
                           unsigned height,
                           float scrwidth,
                           float scrheight)
  : name_(name),
    width_(width),
    height_(height),
    eyedistance_(60.0),
    screenwidth_(scrwidth),
    screenheight_(scrheight)
{
  auto bbox_size = application::instance().bbox().size();
  gpucast::gl::vec3f size(bbox_size[0], bbox_size[1], bbox_size[2]);
  position_camera_ = gpucast::gl::vec3f(0.0, 0.0, size.length());
}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ displaysetup::~displaysetup()
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
displaysetup::resize(unsigned width, unsigned height)
{
  width_  = width;
  height_ = height;
}

///////////////////////////////////////////////////////////////////////////////
unsigned
displaysetup::width() const
{
  return width_;
}

///////////////////////////////////////////////////////////////////////////////
unsigned
displaysetup::height() const
{
  return height_;
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup::camera(float px, float py, float pz, float targetx, float targety, float targetz)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

    // offset x-axis
  float ox(0.0);

  // offset
  float oy(100.0);

  // distance to screen
  float d(2000.0);
  float znear(1.0);
  float zfar(10000.0);

  glFrustum( ( ox - screenwidth_/2 ) * znear / d,( ox + screenwidth_/2 ) * znear / d,( oy - screenheight_/2 ) * znear / d, ( oy + screenheight_/2 ) * znear / d, znear, zfar);
  //gluPerspective(50.0f, float(width())/float(height()), 1.0f, 10000.0f);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  gluLookAt( px,  py,  pz,
             targetx,  targety,  targetz,
             0.0, 1.0, 0.0);
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
displaysetup::increase_eyedistance()
{
  eyedistance_ += 1.0;
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
displaysetup::decrease_eyedistance()
{
  eyedistance_ -= 1.0;
}



///////////////////////////////////////////////////////////////////////////////
// displaysetup_mono
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
displaysetup_mono::displaysetup_mono(unsigned width, unsigned height)
  : displaysetup("Singleframe Setup", width, height, 1480.0f, 830.1f)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */
displaysetup_mono::~displaysetup_mono()
{}

/* virtual */ void
displaysetup_mono::init()
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
displaysetup_mono::display()
{
  application const& app = application::instance();
  gpucast::gl::trackball const& tb = app.trackball();

  // get bounding box of scene
  gpucast::gl::vec3d mn = app.bbox().min;
  gpucast::gl::vec3d mx = app.bbox().max;

  // clear buffer
  glClearColor(app.background()[0], app.background()[1], app.background()[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //glOrtho(-0.5, 0.5, -0.5, 0.5, 0.1, 10.0);
  glOrtho(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  gpucast::gl::matrix4x4<float> const rot = tb.rotation();
  glMultMatrixf(&rot[0]);

  //glTranslatef(0.0, 0.0, -5.0);

  //gpucast::renderer::instance().draw_envmap();

  //camera(0.0, 0.0, 0.0, 0.0, 0.0, -2000.0);
  camera(position_camera_[0],
         position_camera_[1],
         position_camera_[2],
         position_screen_[0],
         position_screen_[1],
         position_screen_[2]);

  // save the initial ModelView matrix before modifying ModelView matrix
  glPushMatrix();
  {
    // transform camera
    glTranslatef(tb.shiftx(), tb.shifty(), tb.distance());

    gpucast::gl::matrix4x4<float> const rot = tb.rotation();
    glMultMatrixf(&rot[0]);

    // object transformation in object coordinates to center object in local coordinate system
    glPushMatrix();
    {
      glTranslatef(float(-(mn[0]+mx[0]) / 2.0),
                   float(-(mn[1]+mx[1]) / 2.0),
                   float(-(mn[2]+mx[2]) / 2.0));

	    //gpucast::renderer::instance().predraw_();
	    application::instance().draw();
	    //gpucast::renderer::instance().postdraw_();

	    // render more geometry

	    // check if z-values are opengl conform
	    /*glColor3f(1.0, 0.0, 0.0);
      glBegin(GL_QUADS);
      {
        glVertex3f(-1000.0, 0.0, -1000.0);
        glVertex3f(-1000.0, 0.0,  1000.0);
        glVertex3f( 1000.0, 0.0,  1000.0);
        glVertex3f( 1000.0, 0.0, -1000.0);
      }
      glEnd();*/

    } glPopMatrix();
  }
  glPopMatrix();

  //gpucast::gl::timer::instance().frame();
}



///////////////////////////////////////////////////////////////////////////////
displaysetup_stereo_karo::displaysetup_stereo_karo(unsigned width, unsigned height)
: displaysetup("Samsung 3DTV", width, height, 1480.0f, 830.1f),
  fbo_(),
  rb_(),
  sb_(),
  program_(0),
  left_(),
  right_()
{
  gpucast::gl::vec3f size = application::instance().bbox().max - application::instance().bbox().min;
  position_camera_ = gpucast::gl::vec3f(0.0, 0.0, size.length());
}

///////////////////////////////////////////////////////////////////////////////
displaysetup_stereo_karo::~displaysetup_stereo_karo()
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */void
displaysetup_stereo_karo::init()
{
 const char* shader_code =
 "#version 330 compatibility \n \
  #extension GL_EXT_gpu_shader4 : enable \n \
   \n \
  uniform sampler2D left;  \n \
  uniform sampler2D right; \n \
   \n \
  void main(void) \n \
  { \n \
    vec2 texCoord = gl_TexCoord[0].xy; \n \
    vec4 pixel_left_eye = texture2D(left,  vec2(texCoord.x, texCoord.y)); \n \
    vec4 pixel_rght_eye = texture2D(right, vec2(texCoord.x, texCoord.y)); \n \
    \n \
    bool rl = mod(gl_FragCoord.x, 2) != mod(gl_FragCoord.y, 2); \n \
    gl_FragColor = mix(pixel_left_eye, pixel_rght_eye, float(rl)); \n \
  }\n";

  program_ = new gpucast::gl::program;
  gpucast::gl::fragmentshader* fprg = new gpucast::gl::fragmentshader;
  fprg->set_source(shader_code);
  fprg->compile();
  program_->add(fprg);
  program_->link();

  initFBO();
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup_stereo_karo::initFBO()
{
  gpucast::gl::texture2d tmp_l;
  gpucast::gl::texture2d tmp_r;

  left_.swap(tmp_l);
  right_.swap(tmp_r);

  initTexture(left_, width(), height());
  initTexture(right_, width(), height());

  fbo_.bind();
  fbo_.attach_texture(left_,  GL_COLOR_ATTACHMENT0_EXT);
  fbo_.attach_texture(right_, GL_COLOR_ATTACHMENT1_EXT);

  rb_.set(GL_DEPTH_COMPONENT32F_NV, width(), height());
  fbo_.attach_renderbuffer(rb_, GL_DEPTH_ATTACHMENT_EXT);

  fbo_.status();

  fbo_.unbind();
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup_stereo_karo::initTexture(gpucast::gl::texture2d& tex, int width, int height)
{
  tex.bind();

  tex.set_parameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  tex.set_parameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  tex.set_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP);
  tex.set_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP);

  tex.teximage(0, GL_RGBA8, width, height, 0, GL_RGB, GL_FLOAT, 0);
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
displaysetup_stereo_karo::display()
{
  // save current draw buffer
  int current_viewport[4];
  GLint current_drawbuffer = 0;
  glGetIntegerv(GL_DRAW_BUFFER, &current_drawbuffer);
  glGetIntegerv(GL_VIEWPORT, current_viewport);

  // set drawbuffer to fbo and render left eye
  fbo_.bind();
  glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

  // clear buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glViewport(0, 0, width(), height());
  camera(position_camera_[0] - eyedistance_/2.0f,
         position_camera_[1],
         position_camera_[2],
         position_screen_[0],
         position_screen_[1],
         position_screen_[2]);
  render();

  // set drawbuffer to fbo and render left eye
  fbo_.bind();
  glDrawBuffer(GL_COLOR_ATTACHMENT1_EXT);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glViewport(0, 0, width(), height());
  camera(position_camera_[0] + eyedistance_/2.0f,
         position_camera_[1],
         position_camera_[2],
         position_screen_[0],
         position_screen_[1],
         position_screen_[2]);
  render();

  fbo_.unbind();

  // restore default drawbuffer
  glDrawBuffer(current_drawbuffer);
  glViewport( current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0f);

  glDisable(GL_LIGHTING);

  program_->begin();

  program_->set_texture2d("left",   left_   , 0);
  program_->set_texture2d("right",  right_  , 1);

  glColor4f(1.0, 1.0, 1.0, 1.0);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  {
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -0.5f);
    glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, -0.5f);
    glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -0.5f);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -0.5f);
  }
  glEnd();
  glDisable(GL_TEXTURE_2D);

  glEnable(GL_LIGHTING);

  program_->end();
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */
void
displaysetup_stereo_karo::resize(unsigned width, unsigned height)
{
  displaysetup::resize(width, height);
  initFBO();
}

///////////////////////////////////////////////////////////////////////////////
void displaysetup_stereo_karo::camera(float px, float py, float pz,
                                      float targetx, float targety, float targetz)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // offset x-axis
  float ox(0.0);

  // offset
  float oy(100.0);

  // distance to screen
  float d(2000.0);
  float znear(300.0);
  float zfar(10000.0);

  glFrustum( ( ox - screenwidth_/2 ) * znear / d,( ox + screenwidth_/2 ) * znear / d,( oy - screenheight_/2 ) * znear / d, ( oy + screenheight_/2 ) * znear / d, znear, zfar);
  //gluPerspective(50.0f, float(width())/float(height()), 1.0f, 10000.0f);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  gluLookAt( px,  py,  pz,
             targetx,  targety,  targetz,
             0.0, 1.0, 0.0);
}


///////////////////////////////////////////////////////////////////////////////
void
displaysetup_stereo_karo::render()
{
  application const& app = application::instance();
  gpucast::gl::trackball const& tb = app.trackball();

  // get bounding box of scene
  gpucast::gl::vec3d mn = app.bbox().min;
  gpucast::gl::vec3d mx = app.bbox().max;

  glMatrixMode(GL_MODELVIEW);

  // save the initial ModelView matrix before modifying ModelView matrix
  glPushMatrix();
  {
    // transform camera
    glTranslatef(tb.shiftx(), tb.shifty(), tb.distance());

    gpucast::gl::matrix4x4<float> const rot = tb.rotation();
    glMultMatrixf(&rot[0]);

    // object transformation in object coordinates to center object in local coordinate system
    glPushMatrix();
    {
      glTranslatef(float(-(mn[0]+mx[0]) / 2.0),
                   float(-(mn[1]+mx[1]) / 2.0),
                   float(-(mn[2]+mx[2]) / 2.0));


	    // draw stuff
	    //gpucast::renderer::instance().predraw();
	    application::instance().draw();
	    //gpucast::renderer::instance().postdraw();

    } glPopMatrix();
  }
  glPopMatrix();
}
