/********************************************************************************
*
* Copyright (C) Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : simple.cpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// local includes
#include <gpucast/gl/atomicbuffer.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/util/contextinfo.hpp>
#include <gpucast/gl/error.hpp>

#include <gpucast/math/matrix4x4.hpp>

#include <gpucast/gl/util/vsync.hpp>
#include <gpucast/gl/util/trackball.hpp>


class application
{
public :

  application()
    : _program(),
    _cube(0, -1, 2, 1),
    _trackball(new gpucast::gl::trackball)
  {
    init_shader(); 

    glEnable(GL_DEPTH_TEST);
  }
 
  ~application()  
  {}   
    
  void init_shader()  
  {
    std::string vertexshader_code = R"(
     #version 420 core
     #extension GL_ARB_separate_shader_objects : enable 
      
      layout (location = 0) in vec4 vertex;   
      layout (location = 1) in vec4 texcoord;   
      layout (location = 2) in vec4 normal;   
      
      uniform mat4 modelviewprojectionmatrix; 
      uniform mat4 modelviewmatrix; 
      uniform mat4 normalmatrix; 
      
      out vec4 fragnormal;  
      out vec4 fragtexcoord;
      out vec4 fragposition;
      
      void main(void) 
      { 
        fragtexcoord = texcoord; 
        fragnormal   = normalmatrix * normal; 
        fragposition = modelviewmatrix * vertex; 
        gl_Position  = modelviewprojectionmatrix * vertex; 
      })"; 

    std::string fragmentshader_code = R"(
       #version 420 core
       #extension GL_ARB_separate_shader_objects : enable

       in vec4 fragnormal;   
       in vec4 fragtexcoord; 
       in vec4 fragposition; 
       
       layout(binding = 3, offset = 0) uniform atomic_uint counter;

       layout (location = 0, index = 0) out vec4 color; 
       
       void main(void) 
       { 
         uint nfrag = atomicCounterIncrement(counter);
         //color = vec4( float(mod(nfrag, 2048) / 2048), float(mod(nfrag, 256) / 256), float(mod(nfrag, 1024) / 1024), 1.0);
         color = vec4( float(nfrag)/65536.0, float(nfrag)/65536.0, float(nfrag)/65536.0, 1.0);
       }
      )";
  
    gpucast::gl::vertexshader   vs;
    gpucast::gl::fragmentshader fs;
 
    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());
    
    vs.compile();
    fs.compile();

    _program.add(&fs);
    _program.add(&vs);

    _program.link();   
  }


  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv   = view * model;
    gpucast::math::matrix4f mvp  = proj * mv;
    gpucast::math::matrix4f nm   = mv.normalmatrix();

    _program.begin();

    _program.set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program.set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program.set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    if (!_counter) {
      _counter = std::make_shared<gpucast::gl::atomicbuffer>(sizeof(unsigned int), GL_DYNAMIC_COPY);
    }

    // initialize buffer with 0
    _counter->bind();
    unsigned* mapped_mem_write = (unsigned*)_counter->map_range(0, sizeof(unsigned), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    *mapped_mem_write = 0;
    _counter->unmap();

    // bind atomic counter buffer and draw
    _counter->bind_buffer_base(3);
    _cube.draw();

    // bind to read number of fragments written
    _counter->bind();
    unsigned* mapped_mem_read = (unsigned*)_counter->map_range(0, sizeof(unsigned), GL_MAP_READ_BIT);

    if (_framecount++ % 50 == 0)
      std::cout << "#Fragments : " << *mapped_mem_read << std::endl;

    _counter->unmap();
    _counter->unbind();

    _program.end();
  }
  
  void resize(int w, int h)
  {
    glViewport(0, 0, w, h);
  }

  void mousebutton(int b, int action)
  {
    gpucast::gl::eventhandler::state state = (action == GLFW_PRESS) ? gpucast::gl::eventhandler::press : gpucast::gl::eventhandler::release;
    gpucast::gl::eventhandler::button button;
    switch (b) {
    case GLFW_MOUSE_BUTTON_1: button = gpucast::gl::eventhandler::left; break;
    case GLFW_MOUSE_BUTTON_2: button = gpucast::gl::eventhandler::right; break;
    case GLFW_MOUSE_BUTTON_3: button = gpucast::gl::eventhandler::middle; break;
    };

    _trackball->mouse(button, state, _currentx, _currenty);
  }

  void motion(double x, double y)
  {
    _currentx = int(x);
    _currenty = int(y);

    _trackball->motion(x, y);
  }

  void keyboard(int key, int action)
  {}

  void reset()
  {}

public :

  int _currentx;
  int _currenty;

  gpucast::gl::program                       _program;
  gpucast::gl::cube                          _cube;
  std::shared_ptr<gpucast::gl::atomicbuffer> _counter;
  std::shared_ptr<gpucast::gl::trackball>    _trackball;
  unsigned                                   _framecount = 0;
};

application* the_app = nullptr;

void
glfw_display()
{
  if (the_app)
    the_app->draw();
}

void
glfw_resize(GLFWwindow* window, int w, int h)
{
  glfwMakeContextCurrent(window);
  if (the_app)
    the_app->resize(w, h);
}

void glfw_mousebutton(GLFWwindow* window, int button, int action, int mods)
{
  glfwMakeContextCurrent(window);
  if (the_app)
    the_app->mousebutton(button, action);
}

void glfw_motion(GLFWwindow* window, double x, double y)
{
  glfwMakeContextCurrent(window);
  if (the_app)
    the_app->motion(x, y);
}

void glfw_key(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  glfwMakeContextCurrent(window);
  if (action == GLFW_RELEASE)
  {
    switch (key) {
      // ESC key
    case 27:
    {
      if (the_app) {
        the_app->reset();
        std::cout << "reset application" << std::endl;
        exit(0);
      }
    }
    break;
    case 'f':
      //toggle_fullscreen();
      break;
    }
  }

  if (the_app)
    the_app->keyboard(key, action);
}

int main(int argc, char** argv)
{
  const int winx = 1024;
  const int winy = 1024;

  glfwInit();

  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);

  GLFWwindow* window = glfwCreateWindow(winx, winy, "atomic_counter", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  glfwSetFramebufferSizeCallback(window, glfw_resize);

  glfwSetKeyCallback(window, glfw_key);
  glfwSetMouseButtonCallback(window, glfw_mousebutton);
  glfwSetCursorPosCallback(window, glfw_motion);

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  glewExperimental = true;
  glewInit();

  gpucast::gl::print_contextinfo(std::cout);

  the_app = new application;

  // somehow initial resize necessary?!?
  glfw_resize(window, winx, winy);

  gpucast::gl::set_vsync(false);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
    glfw_display();
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();

  delete the_app;
}
