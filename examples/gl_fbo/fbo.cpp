#include <GL/glew.h>
#include <GL/freeglut.h>

#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/math/vec4.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/eventhandler.hpp>

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

    gpucast::gl::framebufferobject*  _fbo;
    gpucast::gl::texture2d*          _colortexture;
    gpucast::gl::renderbuffer*       _depthbuffer;

    gpucast::gl::cube*               _cube0;
    gpucast::gl::cube*               _cube1;

    gpucast::gl::program*            _firstpass;
    gpucast::gl::program*            _secondpass;

    gpucast::gl::trackball*          _trackball;
    gpucast::math::matrix4f          _mv;
    gpucast::math::matrix4f          _mp;
    gpucast::math::matrix4f          _mc;

    std::size_t               _screenwidth;
    std::size_t               _screenheight;

    std::size_t               _fbowidth;
    std::size_t               _fboheight;

    float                     _animation;
    gpucast::math::vec4f               _lightpos;

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
      _trackball    ( new gpucast::gl::trackball ),
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

      glewExperimental = true;
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
      glutInitContextVersion  ( 4, 3);
      glutInitContextProfile  ( GLUT_CORE_PROFILE );

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
      _cube0 = new gpucast::gl::cube(0, 3, 1, 2);;
      _cube0->set_color(0.8f, 0.6f, 0.1f);

      _cube1 = new gpucast::gl::cube(0, 3, 1, 2);;
      _cube1->set_color(1.0f, 1.0f, 1.0f);
    }


    ///////////////////////////////////////////////////////////////////////////
    void init_fbo()
    {
      _colortexture = new gpucast::gl::texture2d        ( );
      _colortexture->teximage(0, GL_RGBA, GLsizei(_fbowidth), GLsizei(_fboheight), 0, GL_RGBA, GL_UNSIGNED_INT, 0);
      _colortexture->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      _colortexture->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      _fbo          = new gpucast::gl::framebufferobject( );

      _depthbuffer  = new gpucast::gl::renderbuffer     ( GL_DEPTH_COMPONENT32F_NV, _fbowidth, _fboheight );

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
      gpucast::math::matrix4f mv = gpucast::math::lookat(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f) *
                          gpucast::math::make_rotation_x(_animation) *
                          gpucast::math::make_rotation_y(_animation) *
                          gpucast::math::make_rotation_z(_animation);

      gpucast::math::matrix4f mvp = _mp * mv;
      gpucast::math::matrix4f nm = mv.normalmatrix();

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
    glEnable(gpucast::gl::texture2d::target());

    // render cube again and use previously rendered image as texture
    {
      gpucast::math::matrix4f mv = gpucast::math::lookat(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f) *
                          gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                          _trackball->rotation() *
                          gpucast::math::make_rotation_x(_animation) *
                          gpucast::math::make_rotation_y(_animation) *
                          gpucast::math::make_rotation_z(_animation);

      gpucast::math::matrix4f mvp = _mp * mv;
      gpucast::math::matrix4f nm = mv.normalmatrix();

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
    _firstpass = new gpucast::gl::program;

    std::string fst_vs = R"(
     #version 430 core
     #extension GL_ARB_separate_shader_objects : enable 
     layout (location = 0) in vec4 vertex;    
     layout (location = 1) in vec4 normal;    
     layout (location = 2) in vec4 texcoord;  
     layout (location = 3) in vec4 color;     
     
     layout (location = 4) uniform mat4 mvp;  
     layout (location = 5) uniform mat4 mv;   
     layout (location = 6) uniform mat4 nm;   
     
     out vec4 fragnormal;
     out vec4 fragcolor;
     out vec4 fragtexcoord;
     out vec4 fragvertex;
     
    void main(void) 
    { 
      fragnormal    = nm * normal; 
      fragcolor     = color; 
      fragtexcoord  = texcoord; 
      fragvertex    = mv * vertex; 
      gl_Position   = mvp * vertex; 
    })";

    std::string fst_fs = R"(
     #version 430 core
     #extension GL_ARB_separate_shader_objects : enable 
      in vec4 fragnormal;   
      in vec4 fragtexcoord; 
      in vec4 fragcolor;    
      in vec4 fragvertex;   
      
      layout (location = 1) uniform vec4 light; 
      layout (location = 0) out vec4 out_color; 
      
      void main(void) 
      { 
        vec3 L = normalize(light.xyz - fragvertex.xyz); 
        vec3 N = normalize(fragnormal.xyz); 
        vec4 color = dot(N,L) * fragcolor; 
        
        out_color = vec4(color.xyz, 1.0f); 
      })";

      gpucast::gl::shader vs1(gpucast::gl::vertex_stage);
      gpucast::gl::shader fs1(gpucast::gl::fragment_stage);

      vs1.set_source(fst_vs.c_str());
      fs1.set_source(fst_fs.c_str());

      vs1.compile();
      fs1.compile();

      _firstpass->add(&vs1);
      _firstpass->add(&fs1);

      _firstpass->link();


    _secondpass = new gpucast::gl::program;

    std::string snd_fs = R"(
      #version 430 core
      #extension GL_ARB_separate_shader_objects : enable 
      in vec4 fragnormal;   
      in vec4 fragtexcoord; 
      in vec4 fragcolor;    
      in vec4 fragvertex;   
      
      layout (location = 0) out vec4 out_color; 
      layout (location = 1) uniform vec4 light; 
      layout (location = 2) uniform sampler2D fbo;
      
      void main(void) 
      { 
        vec3 L = normalize(light.xyz - fragvertex.xyz); 
        vec3 N = normalize(fragnormal.xyz); 
        vec2 texcoord; 
        if ( fragtexcoord.x > 0.9999 || fragtexcoord.x < 0.00001 ) texcoord = fragtexcoord.yz; 
        if ( fragtexcoord.y > 0.9999 || fragtexcoord.y < 0.00001 ) texcoord = fragtexcoord.xz; 
        if ( fragtexcoord.z > 0.9999 || fragtexcoord.z < 0.00001 ) texcoord = fragtexcoord.xy; 
        vec4 color = dot(N,L) * fragcolor * texture2D(fbo, texcoord); 
        out_color = vec4(color.xyz, 1.0f); 
      })";

      gpucast::gl::shader vs2(gpucast::gl::vertex_stage);
      gpucast::gl::shader fs2(gpucast::gl::fragment_stage);

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

  the_fbo_test_application->_mp = gpucast::math::perspective(60.0f, float(width)/height, 1.0f, 1000.0f);
}

void fbo_test_application::idleCB()
{
  glutPostRedisplay();
}


void fbo_test_application::mouseCB(int button, int state, int x, int y)
{
  the_fbo_test_application->_trackball->mouse(gpucast::gl::eventhandler::button(button), gpucast::gl::eventhandler::state(state), x, y);
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

