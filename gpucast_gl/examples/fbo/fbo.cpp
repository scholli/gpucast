#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glpp/framebufferobject.hpp>
#include <glpp/renderbuffer.hpp>
#include <glpp/texture2d.hpp>
#include <glpp/arraybuffer.hpp>
#include <glpp/elementarraybuffer.hpp>
#include <glpp/math/matrix4x4.hpp>
#include <glpp/math/vec4.hpp>
#include <glpp/program.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/primitives/cube.hpp>
#include <glpp/error.hpp>
#include <glpp/util/eventhandler.hpp>

#include <vector>

class fbo_test_application;
fbo_test_application* the_fbo_test_application = 0;

float randf()
{
  return float(rand())/RAND_MAX;
}

class fbo_test_application
{
  private : // attributes

    glpp::framebufferobject*  _fbo;
    glpp::texture2d*          _colortexture;
    glpp::renderbuffer*       _depthbuffer;

    glpp::cube*               _cube0;
    glpp::cube*               _cube1;

    glpp::program*            _firstpass;
    glpp::program*            _secondpass;

    glpp::trackball*          _trackball;
    glpp::matrix4f            _mv;
    glpp::matrix4f            _mp;
    glpp::matrix4f            _mc;

    std::size_t               _screenwidth;
    std::size_t               _screenheight;

    std::size_t               _fbowidth;
    std::size_t               _fboheight;

    float                     _animation;
    glpp::vec4f               _lightpos;

  public : // ctor

    fbo_test_application(int argc, char** argv,
                         std::size_t window_width,
                         std::size_t window_height,
                         std::size_t fbo_width,
                         std::size_t fbo_height)
    : _fbo          ( 0 ),
      _colortexture ( 0 ),
      _depthbuffer  ( 0 ),
      _cube0        ( 0 ),
      _cube1        ( 0 ),
      _firstpass    ( 0 ),
      _secondpass   ( 0 ),
      _trackball    ( new glpp::trackball ),
      _mp           ( ),
      _screenwidth  ( window_width ),
      _screenheight ( window_height ),
      _fbowidth     ( fbo_width ),
      _fboheight    ( fbo_height),
      _animation    ( 0.0f ),
      _lightpos     ( 0.0f, 0.0f, 0.0f, 1.0f )
    {
      initGLUT(argc, argv);
      initGL();

      glewInit();

      init_geometry();
      init_fbo();
      init_shader();
    }

  public : // member functions

    ///////////////////////////////////////////////////////////////////////////
    void initGL()
    {
      // enable /disable features
      glEnable(GL_TEXTURE_2D);
      glEnable(GL_DEPTH_TEST);
    }


    ///////////////////////////////////////////////////////////////////////////
    int initGLUT(int argc, char **argv)
    {
      glutInit(&argc, argv);
      glutInitContextVersion  ( 3, 3);
      glutInitContextProfile  ( GLUT_COMPATIBILITY_PROFILE );

      glutInitDisplayMode   ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
      glutInitWindowSize    ( int(_screenwidth), int(_screenheight));
      glutInitWindowPosition( 0, 0);

      int handle = glutCreateWindow(argv[0]);

      // register GLUT callback functions
      glutDisplayFunc ( displayCB );
      glutIdleFunc    ( idleCB );
      glutReshapeFunc ( reshapeCB );
      glutMouseFunc   ( mouseCB );
      glutMotionFunc  ( mouseMotionCB );
      atexit          ( exitCB );

      return handle;
    }


    ///////////////////////////////////////////////////////////////////////////
    void init_geometry()
    {
      _cube0 = new glpp::cube(0, 3, 1, 2);;
      _cube0->set_color(0.8f, 0.6f, 0.1f);

      _cube1 = new glpp::cube(0, 3, 1, 2);;
      _cube1->set_color(1.0f, 1.0f, 1.0f);
    }


    ///////////////////////////////////////////////////////////////////////////
    void init_fbo()
    {
      _colortexture = new glpp::texture2d        ( );
      _colortexture->teximage(0, GL_RGBA, GLsizei(_fbowidth), GLsizei(_fboheight), 0, GL_RGBA, GL_UNSIGNED_INT, 0);
      _colortexture->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      _colortexture->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      _fbo          = new glpp::framebufferobject( );

      _depthbuffer  = new glpp::renderbuffer     ( GL_DEPTH_COMPONENT32F_NV, _fbowidth, _fboheight );

      _fbo->attach_renderbuffer(*_depthbuffer, GL_DEPTH_ATTACHMENT_EXT);
      _fbo->attach_texture(*_colortexture, GL_COLOR_ATTACHMENT0_EXT, 0);
    }


  ///////////////////////////////////////////////////////////////////////////////
  void draw()
  {
#if WIN32
    Sleep(10);
#else
    usleep(10000);
#endif
    

    // do animation
    _animation += 0.01f;

    // set viewport to fbo size
    glViewport(0, 0, GLsizei(_fbowidth), GLsizei(_fboheight));

    // bind frame buffer object
    _fbo->bind();
    //_fbo->print(std::cout);

    // clear attached colorbuffer and depth buffer
    glClearColor(1, 1, 1, 1);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // compute transformation matrices
    {
      glpp::matrix4f mv = glpp::lookat(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f) *
                          glpp::make_rotation_x(_animation) *
                          glpp::make_rotation_y(_animation) *
                          glpp::make_rotation_z(_animation);

      glpp::matrix4f mvp = _mp * mv;
      glpp::matrix4f nm = mv.normalmatrix();

      // bind shader and set uniforms
      _firstpass->begin();
      _firstpass->set_uniform_matrix4fv("mvp", 1, false, &mvp[0]);
      _firstpass->set_uniform_matrix4fv("mv",  1, false, &mv[0]);
      _firstpass->set_uniform_matrix4fv("nm",  1, false, &nm[0]);
      _firstpass->set_uniform4fv("light", 1, &_lightpos[0]);

      // draw cube into fbo
      _cube0->draw();

      // unbind shader
      _firstpass->end();
    }

    // unbind fbo
    _fbo->unbind();

    // reset viewport to window size
    glViewport(0, 0, GLsizei(_screenwidth), GLsizei(_screenheight));

    // clear color, depth and stencilbuffer of original framebuffer
    glClearColor(0.2f, 0.2f, 0.2f, 1);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // enable texturing
    glEnable(glpp::texture2d::target());

    // render cube again and use previously rendered image as texture
    {
      glpp::matrix4f mv = glpp::lookat(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f) *
                          glpp::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                          _trackball->rotation() *
                          glpp::make_rotation_x(_animation) *
                          glpp::make_rotation_y(_animation) *
                          glpp::make_rotation_z(_animation);

      glpp::matrix4f mvp = _mp * mv;
      glpp::matrix4f nm = mv.normalmatrix();

      // bind shader and set uniforms
      _secondpass->begin();
      _secondpass->set_uniform_matrix4fv("mvp", 1, false, &mvp[0]);
      _secondpass->set_uniform_matrix4fv("mv",  1, false, &mv[0]);
      _secondpass->set_uniform_matrix4fv("nm",  1, false, &nm[0]);
      _secondpass->set_uniform4fv("light", 1, &_lightpos[0]);
      _secondpass->set_texture2d("fbo", *_colortexture, 2);

      // draw cube into fbo
      _cube1->draw();

      // unbind shader
      _secondpass->end();
    }

    // swap back and frontbuffer
    glutSwapBuffers();
  }

  void init_shader()
  {
    _firstpass = new glpp::program;

    std::string fst_vs =
    "#version 330 compatibility \n \
     #extension GL_ARB_separate_shader_objects : enable \n \
     layout (location = 0) in vec4 vertex;    \n \
     layout (location = 1) in vec4 normal;    \n \
     layout (location = 2) in vec4 texcoord;  \n \
     layout (location = 3) in vec4 color;     \n \
     \n \
     layout (location = 4) uniform mat4 mvp;  \n \
     layout (location = 5) uniform mat4 mv;   \n \
     layout (location = 6) uniform mat4 nm;   \n \
     \n \
     out vec4 fragnormal;\n \
     out vec4 fragcolor;\n \
     out vec4 fragtexcoord;\n \
     out vec4 fragvertex;\n \
     \n \
    void main(void) \n \
    { \n \
      fragnormal    = nm * normal; \n \
      fragcolor     = color; \n \
      fragtexcoord  = texcoord; \n \
      fragvertex    = mv * vertex; \n \
      gl_Position   = mvp * vertex; \n \
    }\n";

    std::string fst_fs =
    "#version 330 compatibility \n \
     #extension GL_ARB_separate_shader_objects : enable \n \
      in vec4 fragnormal;   \n \
      in vec4 fragtexcoord; \n \
      in vec4 fragcolor;    \n \
      in vec4 fragvertex;   \n \
      \n \
      layout (location = 1) uniform vec4 light; \n \
      layout (location = 0) out vec4 out_color; \n \
      \n \
      void main(void) \n \
      { \n \
        vec3 L = normalize(light.xyz - fragvertex.xyz); \n \
        vec3 N = normalize(fragnormal.xyz); \n \
        vec4 color = dot(N,L) * fragcolor; \n \
        \n \
        out_color = vec4(color.xyz, 1.0f); \n \
      }\n";

      glpp::vertexshader vs1;
      glpp::fragmentshader fs1;

      vs1.set_source(fst_vs.c_str());
      fs1.set_source(fst_fs.c_str());

      vs1.compile();
      fs1.compile();

      _firstpass->add(&vs1);
      _firstpass->add(&fs1);

      _firstpass->link();


    _secondpass = new glpp::program;

    std::string snd_fs =
    "#version 330 compatibility \n \
      #extension GL_ARB_separate_shader_objects : enable \n \
      in vec4 fragnormal;   \n \
      in vec4 fragtexcoord; \n \
      in vec4 fragcolor;    \n \
      in vec4 fragvertex;   \n \
      \n \
      layout (location = 0) out vec4 out_color; \n \
      layout (location = 1) uniform vec4 light; \n \
      layout (location = 2) uniform sampler2D fbo;\n \
      \n \
      void main(void) \n \
      { \n \
        vec3 L = normalize(light.xyz - fragvertex.xyz); \n \
        vec3 N = normalize(fragnormal.xyz); \n \
        vec2 texcoord; \n \
        if ( fragtexcoord.x > 0.9999 || fragtexcoord.x < 0.00001 ) texcoord = fragtexcoord.yz; \n \
        if ( fragtexcoord.y > 0.9999 || fragtexcoord.y < 0.00001 ) texcoord = fragtexcoord.xz; \n \
        if ( fragtexcoord.z > 0.9999 || fragtexcoord.z < 0.00001 ) texcoord = fragtexcoord.xy; \n \
        vec4 color = dot(N,L) * fragcolor * texture2D(fbo, texcoord); \n \
        out_color = vec4(color.xyz, 1.0f); \n \
      }\n";

      glpp::vertexshader vs2;
      glpp::fragmentshader fs2;

      vs2.set_source(fst_vs.c_str());
      fs2.set_source(snd_fs.c_str());

      vs2.compile();
      fs2.compile();

      _secondpass->add(&vs2);
      _secondpass->add(&fs2);

      _secondpass->link();
    }



    ///////////////////////////////////////////////////////////////////////////
    void run ()
    {
      glutMainLoop();
    }

  public : // GLUT callbacks

    static void displayCB    ( );
    static void reshapeCB    ( int w, int h );
    static void idleCB       ( );
    static void mouseCB      ( int button, int stat, int x, int y );
    static void mouseMotionCB( int x, int y );
    static void exitCB       ( );
};




///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void fbo_test_application::displayCB()
{
  the_fbo_test_application->draw();
}


void fbo_test_application::reshapeCB(int width, int height)
{
  // set viewport to be the entire window
  glViewport(0, 0, (GLsizei)width, (GLsizei)height);

  the_fbo_test_application->_screenwidth  = width;
  the_fbo_test_application->_screenheight = height;

  the_fbo_test_application->_mp = glpp::perspective(60.0f, float(width)/height, 1.0f, 1000.0f);
}

void fbo_test_application::idleCB()
{
  glutPostRedisplay();
}


void fbo_test_application::mouseCB(int button, int state, int x, int y)
{
  the_fbo_test_application->_trackball->mouse(glpp::eventhandler::button(button), glpp::eventhandler::state(state), x, y);
}


void fbo_test_application::mouseMotionCB(int x, int y)
{
  the_fbo_test_application->_trackball->motion(x, y);
}

void fbo_test_application::exitCB()
{
  delete(the_fbo_test_application->_fbo);
  delete(the_fbo_test_application->_depthbuffer);
  delete(the_fbo_test_application->_colortexture);
  delete(the_fbo_test_application->_cube0);
  delete(the_fbo_test_application->_cube1);
  delete(the_fbo_test_application->_firstpass);
  delete(the_fbo_test_application->_secondpass);
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
  // create the application
  the_fbo_test_application = new fbo_test_application(argc, argv, 1280, 1024, 512, 512);

  // run it
  the_fbo_test_application->run();

  // destroy it
  delete the_fbo_test_application;

  return 0;
}

