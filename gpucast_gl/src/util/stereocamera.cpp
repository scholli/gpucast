/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : stereo_anaglyph.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// i/f header
#include "gpucast/gl/util/stereocamera.hpp"

#include <iterator>

// header, system
#include <GL/glew.h>
#include <GL/freeglut.h>

// header, project
#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/error.hpp>



namespace gpucast { namespace gl {

  float frand()
  {
    return float(rand())/RAND_MAX;
  }

///////////////////////////////////////////////////////////////////////////////
stereocamera::stereocamera( float eyedistance )
: camera        ( ),
  _eyedistance  ( eyedistance ),
  _fbo          ( ),
  _depthbuffer  ( GL_DEPTH_COMPONENT, 512, 512),
  _anaglyph     ( new program ),
  _checkerboard ( new program ),
  _left         ( ),
  _right        ( )
{
  _init_fbo();
  _init_shader();
}

///////////////////////////////////////////////////////////////////////////////
stereocamera::~stereocamera()
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */
void
stereocamera::resize (std::size_t width, std::size_t height)
{
  camera::resize(width, height);
}

///////////////////////////////////////////////////////////////////////////////
/*virtual*/ void
stereocamera::draw()
{
  if (_drawcallback)
  {
    // backup current draw buffer
    int current_viewport[4];
    GLint current_drawbuffer = 0;
    glGetIntegerv(GL_DRAW_BUFFER, &current_drawbuffer);
    glGetIntegerv(GL_VIEWPORT, current_viewport);

    // set frustum
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glFrustum(( _screenoffset_x - _screenwidth/2  ) * _znear / _screendistance,
              ( _screenoffset_x + _screenwidth/2  ) * _znear / _screendistance,
              ( _screenoffset_y - _screenheight/2 ) * _znear / _screendistance,
              ( _screenoffset_y + _screenheight/2 ) * _znear / _screendistance,
              _znear,
              _zfar);

    // set camera to left eye
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt( _position[0] - _eyedistance/2.0,  _position[1], _position[2],
               _target[0],                   _target[1],   _target[2],
               _up[0],                       _up[1],       _up[2]);

    // bind fbo and draw left eye
    _fbo.bind();

    //_fbo.print(std::cout);
    //_fbo.status();

    _drawcallback();

    // unbind fbo
    _fbo.unbind();

    _left.bind();
    glGenerateMipmapEXT(texture2d::target());
    _left.unbind();

    display(current_drawbuffer, current_viewport);

  }

  glutSwapBuffers();
}



///////////////////////////////////////////////////////////////////////////////
void
stereocamera::display(GLint drawbuffer, GLint viewport[4])
{
  // restore default drawbuffer
  glDrawBuffer ( drawbuffer );
  glViewport   ( viewport[0], viewport[1], viewport[2], viewport[3] );

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, 0.1, 1000.0);

  GLboolean state_light, state_tex2d;
  glGetBooleanv(GL_LIGHTING, &state_light);
  glGetBooleanv(GL_TEXTURE_2D, &state_tex2d);

#if 0
  std::vector<float> v(resolution_x() * resolution_y() * 4);
  std::generate(v.begin(), v.end(), frand);
  _left.teximage(0, 4, resolution_x(), resolution_y(), 0, GL_RGBA, GL_FLOAT, &v.front());
#endif

  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);

  _anaglyph->begin();

  _anaglyph->set_texture2d("left", _left, 0);

  glBegin(GL_QUADS);
  {
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -0.5f);
    glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, -0.5f);
    glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -0.5f);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -0.5f);
  }
  glEnd();

  _anaglyph->end();

  glutSwapBuffers();
}


///////////////////////////////////////////////////////////////////////////////
void
stereocamera::_init_fbo()
{
  _left.set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _left.set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  _left.set_parameteri(GL_TEXTURE_WRAP_T, GL_REPEAT);
  _left.set_parameteri(GL_TEXTURE_WRAP_S, GL_REPEAT);

  _fbo.bind();
  _fbo.attach_texture(_left,  GL_COLOR_ATTACHMENT0_EXT, 0);
  _fbo.attach_renderbuffer(_depthbuffer, GL_DEPTH_ATTACHMENT_EXT);

  _fbo.status();

  _fbo.unbind();
}

///////////////////////////////////////////////////////////////////////////////
float
stereocamera::eyedistance( ) const
{
  return _eyedistance;
}


///////////////////////////////////////////////////////////////////////////////
void
stereocamera::eyedistance( float e)
{
  _eyedistance = e;
}

///////////////////////////////////////////////////////////////////////////////
void
stereocamera::_init_shader()
{
  _init_anaglyph();
  _init_checkerboard();
}


///////////////////////////////////////////////////////////////////////////////
void
stereocamera::_init_anaglyph()
{
  const char* vs_code =
  "#version 330 compatibility \n \
   \n \
  void main(void) \n \
  { \n \
    gl_TexCoord[0] = gl_MultiTexCoord0; \n \
    gl_Position = ftransform(); \n \
  }\n";

 const char* fs_code =
 "#version 330 compatibility \n \
   \n \
  uniform sampler2D left;  \n \
  \n \
  layout(location = 0) out vec4 out_color; \n \
  \n \
  void main(void) \n \
  { \n \
    vec4 lefteye = texture2D(left,  gl_TexCoord[0].xy); \n \
    \n \
    out_color = lefteye; \n \
  }\n";

  fragmentshader* fprg = new fragmentshader;
  vertexshader*   vprg = new vertexshader;

  fprg->set_source(fs_code);
  fprg->compile();

  vprg->set_source(vs_code);
  vprg->compile();

  _anaglyph->add(fprg);
  _anaglyph->add(vprg);

  _anaglyph->link();
}


///////////////////////////////////////////////////////////////////////////////
void
stereocamera::_init_checkerboard()
{}




} } // namespace gpucast / namespace gl
