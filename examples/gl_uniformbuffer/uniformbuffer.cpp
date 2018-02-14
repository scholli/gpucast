/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : simple.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>
#include <cstring>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>

#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/uniformbuffer.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/math/matrix4x4.hpp>



class application
{
public :

  application()
    : _program  (),
      _cube     (0, -1, 2, 1),
      _trackball(new gpucast::gl::trackball),
      _view_ubo(new gpucast::gl::uniformbuffer)
  {
    init_shader(); 

    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));

    _view_ubo->bufferdata(sizeof(matrix_layout), 0, GL_DYNAMIC_DRAW);

    glEnable(GL_DEPTH_TEST);
  }
 
  ~application()  
  {}   
    
  void init_shader()  
  {
    std::string vertexshader_code = R"(
     #version 420 compatibility
     #extension GL_ARB_separate_shader_objects : enable 
      
      layout (location = 0) in vec4 vertex;   
      layout (location = 1) in vec4 texcoord;   
      layout (location = 2) in vec4 normal;   
      
      layout (std140) uniform view_ubo {
        int x;
        int y;
        int z;
        float w;
        mat4 modelviewprojectionmatrix; 
        mat4 modelviewmatrix; 
        mat4 normalmatrix; 
      };
      
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
      #version 420 compatibility
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
      })";
  
    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);
 
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
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math:: make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);

    _view_data.modelviewmatrix = view * model;
    _view_data.modelviewprojectionmatrix = proj * _view_data.modelviewmatrix;
    _view_data.normalmatrix = _view_data.modelviewmatrix.normalmatrix();

    // there are different 
#if 0
    matrix_layout* ubo_address = reinterpret_cast<matrix_layout*>(_view_ubo->map(GL_WRITE_ONLY));
    *ubo_address = _view_data;
    _view_ubo->unmap();
#endif

#if 1
    auto host_address = _view_ubo->map(GL_WRITE_ONLY);
    std::memcpy(host_address, &_view_data, sizeof(matrix_layout));
    _view_ubo->unmap();
#endif

#if 0
    _view_ubo->buffersubdata(0, sizeof(matrix_layout), (void*)&_view_data);
#endif

    _program.set_uniformbuffer("view_ubo", *_view_ubo, 4);

    _program.begin();

    _cube.draw();

    _program.end();
  }
  

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


public :

  gpucast::gl::program                        _program;
  gpucast::gl::cube                           _cube;
  std::shared_ptr<gpucast::gl::trackball>     _trackball;

  struct matrix_layout {
    int x;
    int y;
    int z;
    float w;
    gpucast::math::matrix4f modelviewprojectionmatrix;
    gpucast::math::matrix4f modelviewmatrix;
    gpucast::math::matrix4f normalmatrix;
  };

  matrix_layout                               _view_data;
  std::shared_ptr<gpucast::gl::uniformbuffer> _view_ubo;
};


int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 4, 3, false);

  glewExperimental = true;
  glewInit();

  application app;
  app.run();

  return 0;
}
