/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : texturearray.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/texturearray.hpp>
#include <gpucast/gl/program.hpp>

#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/sampler.hpp>

#include <gpucast/gl/primitives/plane.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/gl/error.hpp>

class application
{
public :
  enum mode_t {first, second, both};

public :

  application(int argc, char* argv[], int x, int y, int width, int height)
    : _trackball(new gpucast::gl::trackball),
      _mode     (both)
  {
    gpucast::gl::glutwindow::init(argc, argv, width, height, x, y);

    glewInit();

    init_textures();
    init_shader();
    _plane = std::shared_ptr<gpucast::gl::plane>(new gpucast::gl::plane(0, 2, 1));

    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);
    gpucast::gl::glutwindow::instance().add_keyevent    ('m', boost::bind(&application::mode, boost::ref(*this), _1, _2));

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));
  }

  ~application()
  {}


  void init_textures() 
  {
    _texture.reset( new gpucast::gl::texturearray );
    _sampler.reset( new gpucast::gl::sampler );

    std::vector<gpucast::math::vec4f> first  (64*64, gpucast::math::vec4f(1.0, 1.0, 0.0, 1.0));
    std::vector<gpucast::math::vec4f> second (64*64, gpucast::math::vec4f(0.5, 0.5, 0.5, 1.0));

#if 1
    _texture->teximage(0, GL_RGBA8, 64, 64, 2, 0, GL_RGBA, GL_FLOAT, 0);
    _texture->texsubimage(0, 0, 0, 0, 64, 64, 1, GL_RGBA, GL_FLOAT, &first.front());
    _texture->texsubimage(0, 0, 0, 1, 64, 64, 1, GL_RGBA, GL_FLOAT, &second.front());
#else
    std::vector<gpucast::math::vec4f> data (64*64*2, gpucast::math::vec4f(1.0, 1.0, 1.0, 1.0));
    _texture->teximage(0, GL_RGBA8, 64, 64, 2, 0, GL_RGBA, GL_FLOAT, &data.front());
#endif
  }


  void init_shader()
  {
    _program = std::shared_ptr<gpucast::gl::program>(new gpucast::gl::program);

    std::string vertexshader_code = R"(
      #version 430 core
      #extension EXT_gpu_shader4 : enable 
      
      layout (location = 0) in vec4 vertex;   
      layout (location = 1) in vec4 texcoord; 
      layout (location = 2) in vec4 normal;   
      
      uniform mat4 modelviewprojectionmatrix; 
      uniform mat4 modelviewmatrix; 
      uniform mat4 normalmatrix; 
      
      out vec4 fragnormal;  
      out vec4 fragtexcoord;
      out vec4 frag_mv;     
      
      void main(void) 
      { 
        fragtexcoord = texcoord; 
        frag_mv      = modelviewmatrix * vertex; 
        fragnormal   = normalmatrix * normal; 
        gl_Position  = modelviewprojectionmatrix * vertex; 
      })";

    std::string fragmentshader_code = R"(
      #version 430 core
      #extension EXT_gpu_shader4 : enable 
      
      uniform sampler2DArray texture; 
      
      in vec4 fragnormal;   
      in vec4 fragtexcoord; 
      in vec4 frag_mv;      
      uniform int mode;     
      
      void main(void) 
      { 
        vec4 tex1    = texture2DArray(texture, vec3(fragtexcoord.xy, 0.5)); 
        vec4 tex2    = texture2DArray(texture, vec3(fragtexcoord.xy, 1)); 
        
        if (mode == 0) gl_FragColor = tex1; 
        if (mode == 1) gl_FragColor = tex2; 
        if (mode == 2) gl_FragColor = fragtexcoord; 
      })";

    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);

    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());
    
    vs.compile();
    fs.compile();

    _program->add(&fs);
    _program->add(&vs);

    _program->link();
  }


  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glEnable(gpucast::gl::texturearray::target());

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv   = view * model;
    gpucast::math::matrix4f mvp  = proj * mv;
    gpucast::math::matrix4f nm   = mv.normalmatrix();

    _program->begin();
  
    _program->set_texturearray("texture", *_texture.get(), 0);

    _program->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program->set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program->set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    switch(_mode) {
      case first :  _program->set_uniform1i("mode", 0); break;
      case second : _program->set_uniform1i("mode", 1); break;
      case both :   _program->set_uniform1i("mode", 2); break;
    };
    _sampler->bind(0);
    _sampler->bind(1);
    _sampler->bind(2);

    _plane->draw();

    _program->end();
  }
  

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


  void mode (int, int)
  {
    switch(_mode) {
      case first :  _mode = second; break;
      case second : _mode = both;   break;
      case both :   _mode = first;  break;
    };
  }


public :

  std::shared_ptr<gpucast::gl::texturearray>   _texture;

  std::shared_ptr<gpucast::gl::program>        _program;
  std::shared_ptr<gpucast::gl::plane>          _plane;
  std::shared_ptr<gpucast::gl::sampler>        _sampler;

  std::shared_ptr<gpucast::gl::trackball>      _trackball;
  mode_t                                  _mode;

};


int main(int argc, char** argv)
{
  application app(argc, argv, 100, 100, 1024, 1024);

  app.run();

  return 0;
}


