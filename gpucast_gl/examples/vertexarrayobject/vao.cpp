/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : vao.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>

// local includes
#include <glpp/glut/window.hpp>

#include <glpp/program.hpp>
#include <glpp/util/camera.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/primitives/cube.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/math/matrix4x4.hpp>

#include <glpp/arraybuffer.hpp>
#include <glpp/vertexarrayobject.hpp>




class application
{
public :

  application()
    : _program  (),
      _trackball(new glpp::trackball),
      _camera   (),
      _vao      (),
      _attrib0  (),
      _attrib1  (),
      _attrib2  ()
  {
    init_shader ();
    init_data   ();

    glpp::glutwindow::instance().add_eventhandler(_trackball);

    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);

    glEnable(GL_DEPTH_TEST);
  }

  ~application()
  {}

  void init_shader()
  {
    std::string vertexshader_code =
    "#version 330 compatibility\n \
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
      out vec4 fragposition;\n \
      \n \
      void main(void) \n \
      { \n \
        fragtexcoord = texcoord; \n \
        fragnormal   = normalmatrix * normal; \n \
        fragposition = modelviewmatrix * vertex; \n \
        gl_Position  = modelviewprojectionmatrix * vertex; \n \
      }\n";

    std::string fragmentshader_code = 
     "#version 330 compatibility\n \
      \n \
      in vec4 fragnormal;   \n \
      in vec4 fragposition; \n \
      in vec4 fragtexcoord; \n \
      \n \
      layout (location = 0) out vec4 color; \n \
      void main(void) \n \
      { \n \
        vec3 V = normalize(-fragposition.xyz);  \n \
        vec3 N = normalize(fragnormal.xyz); \n \
        float attenuation = min(1.0, 10.0 / length(fragposition.xyz)); \n \
        color = attenuation * vec4(dot(V, N)) * 0.3 * fragtexcoord; \n \
      }\n";

    glpp::vertexshader   vs;
    glpp::fragmentshader fs;

    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());
    
    vs.compile();
    fs.compile();

    _program.add(&fs);
    _program.add(&vs);

    std::cout << vs.log() << std::endl;
    std::cout << fs.log() << std::endl;

    _program.link();
  }
  
  void init_data()
  {
    float vertexdata[] = {-1.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f,
                           1.0f, 0.0f, 0.0f };
    float texcoord[]  = {  0.0f, 0.0f, 0.0f,
                           1.0f, 0.0f, 0.0f,
                           1.0f, 1.0f, 0.0f };
    float normal[]    = {  0.0f, 0.0f, 1.0f,
                           0.0f, 0.0f, 1.0f,
                           0.0f, 0.0f, 1.0f };
  
#if 0
    // reserve and copy
    _attrib0.bufferdata(sizeof(vertexdata), vertexdata);
    _attrib1.bufferdata(sizeof(texcoord),   texcoord);
    _attrib2.bufferdata(sizeof(normal),     normal);
#else 
    // reserve and then copy manually by mapping
    std::size_t reserved = 1024;

    _attrib0.bufferdata(reserved, 0);
    _attrib1.bufferdata(reserved, 0);
    _attrib2.bufferdata(reserved, 0);

    std::vector<glpp::vec3f> v0;
    v0.push_back(glpp::vec3f(-1.0f, 0.0f, 0.0f));
    v0.push_back(glpp::vec3f( 0.0f, 1.0f, 0.0f));
    v0.push_back(glpp::vec3f( 2.0f, 0.0f, 0.0f));

    std::copy(v0.begin(), v0.end(), static_cast<glpp::vec3f*>(_attrib0.map(GL_WRITE_ONLY)));        // using STL
    std::copy(&texcoord[0],   &texcoord[8],   static_cast<float*>(_attrib1.map(GL_WRITE_ONLY)));    // using static array
    std::copy(&normal[0],     &normal[8],     static_cast<float*>(_attrib2.map(GL_WRITE_ONLY)));

    _attrib0.unmap();
    _attrib1.unmap();
    _attrib2.unmap();
#endif
    _vao.bind();

    _vao.attrib_array   (_attrib0, 0, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (0);

    _vao.attrib_array   (_attrib1, 1, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (1);

    _vao.attrib_array   (_attrib2, 2, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (2);

    _vao.unbind();
/*
	  glGenVertexArrays(1, &_vao);
	  glBindVertexArray(_vao);
  	
	  glGenBuffers(3, &_va[0]);
  	
	  //glBindBuffer(GL_ARRAY_BUFFER, _va[0]);
	  glNamedBufferDataEXT(_va[0], sizeof(vertexdata), vertexdata, GL_STATIC_DRAW);
    glVertexArrayVertexAttribOffsetEXT(_vao, _va[0], 0, 3, GL_FLOAT, false, 0, 0);
    glEnableVertexArrayAttribEXT(_vao, 0);

	  //glBindBuffer(GL_ARRAY_BUFFER, _va[1]);
	  glNamedBufferDataEXT(_va[1], sizeof(texcoord), texcoord, GL_STATIC_DRAW);
    glVertexArrayVertexAttribOffsetEXT(_vao, _va[1], 1, 3, GL_FLOAT, false, 0, 0);
    glEnableVertexArrayAttribEXT(_vao, 1);

	  //glBindBuffer(GL_ARRAY_BUFFER, _va[2]);
	  glNamedBufferDataEXT(_va[2], sizeof(normal), normal, GL_STATIC_DRAW);
    glVertexArrayVertexAttribOffsetEXT(_vao, _va[2], 2, 3, GL_FLOAT, false, 0, 0);
    glEnableVertexArrayAttribEXT(_vao, 2);*/
  }
    

  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    glpp::matrix4f model = glpp::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    glpp::matrix4f proj = glpp::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    glpp::matrix4f mv   = view * model;
    glpp::matrix4f mvp  = proj * mv;
    glpp::matrix4f nm   = mv.normalmatrix();

    _program.begin();
  
    _program.set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program.set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program.set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    _vao.bind();
	  glDrawArrays(GL_TRIANGLES, 0, 3);
    _vao.unbind();

    _program.end();
  }
  

  void run() 
  {
    glpp::glutwindow::instance().run();
  }


public :

  glpp::program                       _program;
  glpp::camera                        _camera;
  boost::shared_ptr<glpp::trackball>  _trackball;

  glpp::vertexarrayobject             _vao;
  glpp::arraybuffer                   _attrib0;
  glpp::arraybuffer                   _attrib1;
  glpp::arraybuffer                   _attrib2;
};


int main(int argc, char** argv)
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0);
  glewInit();

  application app;
  app.run();

  return 0;
}
