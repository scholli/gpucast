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
#include <glpp/glut/window.hpp>

#include <glpp/texturearray.hpp>
#include <glpp/program.hpp>
#include <glpp/util/camera.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/sampler.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/primitives/plane.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/math/matrix4x4.hpp>
#include <glpp/error.hpp>

class application
{
public :
  enum mode_t {first, second, both};

public :

  application(int argc, char* argv[], int x, int y, int width, int height)
    : _trackball(new glpp::trackball),
      _camera   (),
      _mode     (both)
  {
    glpp::glutwindow::init(argc, argv, width, height, x, y);

    glewInit();

    init_textures();
    init_shader();
    _plane = boost::shared_ptr<glpp::plane>(new glpp::plane(0, 2, 1));

    glpp::glutwindow::instance().add_eventhandler(_trackball);
    glpp::glutwindow::instance().add_keyevent    ('m', boost::bind(&application::mode, boost::ref(*this), _1, _2));

    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);
  }

  ~application()
  {}


  void init_textures() 
  {
    _texture.reset( new glpp::texturearray );
    _sampler.reset( new glpp::sampler );

    std::vector<glpp::vec4f> first  (64*64, glpp::vec4f(1.0, 1.0, 0.0, 1.0));
    std::vector<glpp::vec4f> second (64*64, glpp::vec4f(0.5, 0.5, 0.5, 1.0));

#if 1
    _texture->teximage(0, GL_RGBA8, 64, 64, 2, 0, GL_RGBA, GL_FLOAT, 0);
    _texture->texsubimage(0, 0, 0, 0, 64, 64, 1, GL_RGBA, GL_FLOAT, &first.front());
    _texture->texsubimage(0, 0, 0, 1, 64, 64, 1, GL_RGBA, GL_FLOAT, &second.front());
#else
    std::vector<glpp::vec4f> data (64*64*2, glpp::vec4f(1.0, 1.0, 1.0, 1.0));
    _texture->teximage(0, GL_RGBA8, 64, 64, 2, 0, GL_RGBA, GL_FLOAT, &data.front());
#endif
  }


  void init_shader()
  {
    _program = boost::shared_ptr<glpp::program>(new glpp::program);

    std::string vertexshader_code =
    "#version 330 compatibility\n \
    #extension EXT_gpu_shader4 : enable \n \
      \n \
      layout (location = 0) in vec4 vertex;   \n \
      layout (location = 1) in vec4 texcoord; \n \
      layout (location = 2) in vec4 normal;   \n \
      \n \
      uniform mat4 modelviewprojectionmatrix; \n \
      uniform mat4 modelviewmatrix; \n \
      uniform mat4 normalmatrix; \n \
      \n \
      out vec4 fragnormal;  \n \
      out vec4 fragtexcoord;\n \
      out vec4 frag_mv;     \n \
      \n \
      void main(void) \n \
      { \n \
        fragtexcoord = texcoord; \n \
        frag_mv      = modelviewmatrix * vertex; \n \
        fragnormal   = normalmatrix * normal; \n \
        gl_Position  = modelviewprojectionmatrix * vertex; \n \
      }\n";

    std::string fragmentshader_code = 
     "#version 330 compatibility\n \
      #extension EXT_gpu_shader4 : enable \n \
      \n \
      uniform sampler2DArray texture; \n \
      \n \
      in vec4 fragnormal;   \n \
      in vec4 fragtexcoord; \n \
      in vec4 frag_mv;      \n \
      uniform int mode;     \n \
      \n \
      void main(void) \n \
      { \n \
        vec4 tex1    = texture2DArray(texture, vec3(fragtexcoord.xy, 0.5)); \n \
        vec4 tex2    = texture2DArray(texture, vec3(fragtexcoord.xy, 1)); \n \
        \n \
        if (mode == 0) gl_FragColor = tex1; \n \
        if (mode == 1) gl_FragColor = tex2; \n \
        if (mode == 2) gl_FragColor = fragtexcoord; \n \
      }\n";

    glpp::vertexshader*   vs = new glpp::vertexshader;
    glpp::fragmentshader* fs = new glpp::fragmentshader;

    vs->set_source(vertexshader_code.c_str());
    fs->set_source(fragmentshader_code.c_str());
    
    vs->compile();
    fs->compile();

    _program->add(fs);
    _program->add(vs);

    _program->link();

    delete fs;
    delete vs;
  }


  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glEnable(glpp::texturearray::target());

    glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    glpp::matrix4f model = glpp::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    glpp::matrix4f proj = glpp::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    glpp::matrix4f mv   = view * model;
    glpp::matrix4f mvp  = proj * mv;
    glpp::matrix4f nm   = mv.normalmatrix();

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

    //glpp::error("B");

    _program->end();
  }
  

  void run() 
  {
    glpp::glutwindow::instance().run();
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

  boost::shared_ptr<glpp::texturearray>   _texture;

  boost::shared_ptr<glpp::program>        _program;
  boost::shared_ptr<glpp::plane>          _plane;
  boost::shared_ptr<glpp::sampler>        _sampler;

  boost::shared_ptr<glpp::trackball>      _trackball;
  glpp::camera                            _camera;
  mode_t                                  _mode;

};


int main(int argc, char** argv)
{
  application app(argc, argv, 100, 100, 1024, 1024);

  app.run();

  return 0;
}


