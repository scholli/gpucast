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

#include <GL/glew.h>
#include <GL/freeglut.h>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/timer_query.hpp>
#include <gpucast/math/matrix4x4.hpp>



class application
{
public :

  application()
    : _program  (),
      _cube     (0, -1, 2, 1),
      _trackball(new gpucast::gl::trackball)
  {
    init_shader(); 

    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);

    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));

    glEnable(GL_DEPTH_TEST);
  }
 
  ~application()  
  {}   
    
  void init_shader()  
  {
    std::string vertexshader_code =
    "#version 420 core\n \
     #extension GL_ARB_separate_shader_objects : enable \n \
      \n \
      layout (location = 0) in vec4 vertex;   \n \
      layout (location = 1) in vec4 texcoord;   \n \
      layout (location = 2) in vec4 normal;   \n \
      \n \
      uniform mat4 modelviewprojectionmatrix; \n \
      uniform mat4 modelviewmatrix; \n \
      uniform mat4 normalmatrix; \n \
      \n \
      out vec4 fragnormal;  \n \
      out vec4 fragtexcoord;\n \
      out vec4 fragposition;\n \
      \n \
      void main(void) \n \
      { \n \
        fragtexcoord = texcoord; \n \
        fragnormal   = normalmatrix * normal; \n \
        fragposition = modelviewmatrix * vertex; \n \
        gl_Position  = modelviewprojectionmatrix * vertex; \n \
      }\n"; 

    std::string fragmentshader_code = R"(
       #version 420 core
       #extension GL_ARB_separate_shader_objects : enable

       in vec4 fragnormal;   
       in vec4 fragtexcoord; 
       in vec4 fragposition; 
       
       layout (location = 0) out vec4 color; 
       
       void main(void) 
       { 
         vec3 V = normalize(-fragposition.xyz); 
         vec3 N = normalize(fragnormal.xyz); 
         float attenuation = min(1.0, 10.0 / length(fragposition.xyz));
         color = attenuation * vec4(1.0) * dot(N,V)  +  0.1 * fragtexcoord; 
         color.w = 0.4; 
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

    std::cout << "vertex shader log : " << vs.log() << std::endl;
    std::cout << "fragment shader log : " << fs.log() << std::endl;

    _program.link();   
  }


  void draw()
  {
    _query.begin();

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

    _cube.draw();

    _program.end();
    _query.end();

    // query draw time
    std::cout << "fps: " << double(1000) / _query.result_wait() << "\r";
  }
  

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


public :

  gpucast::gl::program                     _program;
  gpucast::gl::cube                        _cube;
  std::shared_ptr<gpucast::gl::trackball>  _trackball;
  gpucast::gl::timer_query                 _query;
};


int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 4, 2, true);

  glewExperimental = true;
  glewInit();

  application app;
  app.run();

  return 0;
}
