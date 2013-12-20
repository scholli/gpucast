/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : window.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/gl/glut/window.hpp"

// header, system
#include <cassert> // assert
#include <functional>

#include <boost/bind.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

// header, project
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/util/camera.hpp>

namespace gpucast { namespace gl {

  glutwindow* g_window = 0;

////////////////////////////////////////////////////////////////////////////////
glutwindow::glutwindow( int           argc,
                        char*         argv[],
                        std::size_t   width,
                        std::size_t   height,
                        std::size_t   posx,
                        std::size_t   posy,
                        std::size_t   context_major,
                        std::size_t   context_minor,
                        bool          core_profile )
  : _width          ( width ),
    _height         ( height ),
    _posx           ( posx ),
    _posy           ( posy ),
    _keymap         ( ),
    _eventhandler   ( ),
    _camera         ( ),
    _handle         ( 0 ),
    _contextmajor   ( context_major ),
    _contextminor   ( context_minor ),
    _core           ( core_profile )
{
  g_window = this;

  _handle = _init(argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
glutwindow::~glutwindow()
{}


////////////////////////////////////////////////////////////////////////////////
/*static*/ glutwindow&
glutwindow::instance()
{
  assert(g_window);
  return *g_window;
}

////////////////////////////////////////////////////////////////////////////////
/*static*/ void
glutwindow::init(int          argc,
                 char*        argv[],
                 std::size_t  width,
                 std::size_t  height,
                 std::size_t  posx,
                 std::size_t  posy,
                 std::size_t  major,
                 std::size_t  minor,
                 bool         core )
{
  static glutwindow instance(argc, argv, width, height, posx, posy, major, minor, core);
}


////////////////////////////////////////////////////////////////////////////////
void
glutwindow::run()
{
  glutMainLoop();
}


////////////////////////////////////////////////////////////////////////////////
void         
glutwindow::leave ( std::function<void()> f ) const
{
  f();
  glutLeaveMainLoop();
}


///////////////////////////////////////////////////////////////////////////////
std::size_t
glutwindow::width() const
{
  return _width;
}


///////////////////////////////////////////////////////////////////////////////
std::size_t
glutwindow::height() const
{
  return _height;
}


///////////////////////////////////////////////////////////////////////////////
std::size_t
glutwindow::posx() const
{
  return _posx;
}


///////////////////////////////////////////////////////////////////////////////
std::size_t
glutwindow::posy() const
{
  return _posy;
}


///////////////////////////////////////////////////////////////////////////////
void
glutwindow::setcamera ( std::shared_ptr<camera> const& c )
{
  _camera = c;
}


///////////////////////////////////////////////////////////////////////////////
void
glutwindow::add_keyevent( unsigned char key, keyfunction_t event )
{
  _keymap.insert(std::make_pair(key, event));
}


///////////////////////////////////////////////////////////////////////////////
void
glutwindow::add_eventhandler ( eventhandler_ptr eh)
{
  _eventhandler.insert(eh);
}

///////////////////////////////////////////////////////////////////////////////
void
glutwindow::remove_eventhandler ( eventhandler_ptr eh)
{
  _eventhandler.erase(eh);
}


///////////////////////////////////////////////////////////////////////////////
void
glutwindow::printinfo ( std::ostream& os ) const
{
  char* gl_version   = (char*)glGetString(GL_VERSION);
  char* glsl_version = (char*)glGetString(GL_SHADING_LANGUAGE_VERSION);

  GLint context_profile;
  glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &context_profile);

  os << "OpenGL Version String : " << gl_version << std::endl;
  os << "GLSL Version String   : " << glsl_version << std::endl;

  switch (context_profile) {
    case GL_CONTEXT_CORE_PROFILE_BIT :
      os << "Core Profile" << std::endl; break;
    case GL_CONTEXT_COMPATIBILITY_PROFILE_BIT :
      os << "Compatibility Profile" << std::endl; break;
    default :
      os << "Unknown Profile" << std::endl;
  };
}


///////////////////////////////////////////////////////////////////////////////
int
glutwindow::_init( int argc, char* argv[])
{
  // window configuration
  glutInit                ( &argc, argv );

  // init context
  glutInitContextVersion  ( GLint(_contextmajor), GLint(_contextminor));

  if (_core) {
    glutInitContextProfile  ( GLUT_CORE_PROFILE );
  } else {
    glutInitContextProfile  ( GLUT_COMPATIBILITY_PROFILE );
  }

  // configure visual
  glutInitDisplayMode     ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL );
  glutInitWindowSize      ( GLint(_width), GLint(_height) );
  glutInitWindowPosition  ( GLint(_posx), GLint(_posy) );

  // create window
  int handle = glutCreateWindow(argv[0]);

  // register GLUT callback functions
  glutDisplayFunc         ( glutwindow::display );
  glutIdleFunc            ( glutwindow::idle );
  glutReshapeFunc         ( glutwindow::reshape );
  glutKeyboardFunc        ( glutwindow::keyboard );
  glutMouseFunc           ( glutwindow::mouse );
  glutMotionFunc          ( glutwindow::motion );
  glutPassiveMotionFunc   ( glutwindow::passivemotion );

  // add some key events
  _keysetup();

  return handle;
}


///////////////////////////////////////////////////////////////////////////////
void
glutwindow::_keysetup( )
{
  int const ESC_KEY = 27;

  _keymap.insert(std::make_pair( ESC_KEY,  std::bind(std::mem_fn(&glutwindow::_quit)      , this, 0, 0)));
  _keymap.insert(std::make_pair( 'q',      std::bind(std::mem_fn(&glutwindow::_quit)      , this, 0, 0)));
  _keymap.insert(std::make_pair( 'f',      std::bind(std::mem_fn(&glutwindow::_fullscreen), this, 0, 0)));
}


///////////////////////////////////////////////////////////////////////////////
void glutwindow::_fullscreen( int /* x */, int /* y */ )
{
  // check if currently fullscreen is enabled
  if ( glutGet(GLUT_FULL_SCREEN) )
  {
    // leave fullscreen mode
    glutReshapeWindow (GLint(_width), GLint(_height));
    glutPositionWindow(GLint(_posx), GLint(_posy));
  } else {
    // enter fullscreen mode
    glutFullScreen();
  }
}


///////////////////////////////////////////////////////////////////////////////
void
glutwindow::_quit( int /* x */ , int /* y */ )
{
  exit(EXIT_SUCCESS);
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// CALLBACKS //////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/* static */ void
glutwindow::display()
{
  if (glutwindow::instance()._camera)
  {
    glutwindow::instance()._camera->draw();
  }
  else
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glutSwapBuffers();
  }
}


///////////////////////////////////////////////////////////////////////////////
/* static */ void
glutwindow::reshape(int w, int h)
{
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);

  if (!glutGet(GLUT_FULL_SCREEN))
  {
    glutwindow::instance()._width  = w;
    glutwindow::instance()._height = h;
  }

  if (glutwindow::instance()._camera) {
    glutwindow::instance()._camera->resize(w, h);
  }

}


///////////////////////////////////////////////////////////////////////////////
/* static */ void
glutwindow::idle()
{
  glutPostRedisplay();
}


///////////////////////////////////////////////////////////////////////////////
/* static */ void
glutwindow::keyboard(unsigned char key, int x, int y)
{
  // forward key event to all eventhandlers
  std::for_each(glutwindow::instance()._eventhandler.begin(),
                glutwindow::instance()._eventhandler.end(),
                std::bind(std::mem_fn(&eventhandler::keyboard), std::placeholders::_1, key, x, y));

  // handle own key events
  std::map<char, keyfunction_t>::const_iterator i = glutwindow::instance()._keymap.find(key);
  if (i != glutwindow::instance()._keymap.end())
  {
    i->second(x,y);
  }
}


///////////////////////////////////////////////////////////////////////////////
/*static*/ void
glutwindow::mouse(int button, int state, int x, int y)
{
  enum eventhandler::button b;
  enum eventhandler::state  s;

  switch (button) {
    case GLUT_LEFT_BUTTON   : b = eventhandler::left; break;
    case GLUT_RIGHT_BUTTON  : b = eventhandler::right; break;
    case GLUT_MIDDLE_BUTTON : b = eventhandler::middle; break;
    default : return;
  }

  switch (state) {
    case GLUT_DOWN   : s = eventhandler::press;    break;
    case GLUT_UP     : s = eventhandler::release;  break;
    default : return;
  }

  std::for_each(glutwindow::instance()._eventhandler.begin(),
                glutwindow::instance()._eventhandler.end(),
                std::bind(std::mem_fn(&eventhandler::mouse), std::placeholders::_1, b, s, x, y));
}


///////////////////////////////////////////////////////////////////////////////
/*static*/ void
glutwindow::motion(int x, int y)
{
  std::for_each(glutwindow::instance()._eventhandler.begin(),
                glutwindow::instance()._eventhandler.end(),
                std::bind(std::mem_fn(&eventhandler::motion), std::placeholders::_1, x, y));
}


///////////////////////////////////////////////////////////////////////////////
/*static*/ void
glutwindow::passivemotion ( int x, int y)
{
  std::for_each(glutwindow::instance()._eventhandler.begin(),
                glutwindow::instance()._eventhandler.end(),
                std::bind(std::mem_fn(&eventhandler::passivemotion), std::placeholders::_1, x, y));
}

} } // namespace gpucast / namespace gl
