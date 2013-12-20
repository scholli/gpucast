/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : parametric_surface.cpp
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
#include <glpp/error.hpp>
#include <glpp/vertexarrayobject.hpp>
#include <glpp/elementarraybuffer.hpp>

#include <tml/parametric/beziersurface.hpp>
#include <tml/parametric/algorithm/horner.hpp>
#include <tml/parametric/point.hpp>




class application
{
public :

  application()
    : _program  (),
      _camera   (),
      _trackball(new glpp::trackball),
      _vao      (),
      _attrib0  (),
      _attrib1  (),
      _attrib2  (),
      _indices  (),
      _samples  (256)
  {
    init_shader ();
    init_data   ();

    glpp::glutwindow::instance().add_eventhandler(_trackball);

    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
  }

  ~application()
  {}

  void init_shader()
  {
    std::string vertexshader_code =
    "#version 410 core\n \
      \n \
      layout (location = 0) in vec3 vertex;   \n \
      layout (location = 1) in vec3 color;    \n \
      layout (location = 2) in vec3 normal;    \n \
      \n \
      uniform mat4 modelviewprojectionmatrix; \n \
      uniform mat4 modelviewmatrix; \n \
      uniform mat4 normalmatrix; \n \
      \n \
      out vec4 fragposition;\n \
      out vec4 fragcolor;\n \
      out vec4 fragnormal;\n \
      \n \
      void main(void) \n \
      { \n \
        fragposition = modelviewmatrix * vec4(vertex, 1.0); \n \
        \n \
        if ( color.x > 1.0 || color.x < 0.0 || \n \
             color.y > 1.0 || color.y < 0.0 ) \n \
        { \n \
          fragcolor    = vec4(clamp(color.xy, vec2(0.0), vec2(1.0)), 1.0, 1.0); \n \
        } else { \n \
          fragcolor    = vec4(color.xy, 0.0, 1.0); \n \
        } \n \
        fragnormal   = normalmatrix * vec4(normal, 0.0); \n \
        gl_Position  = modelviewprojectionmatrix * vec4(vertex, 1.0); \n \
      }\n";

    std::string fragmentshader_code =
     "#version 410 core \n \
      \n \
      in vec4 fragposition;\n \
      in vec4 fragcolor;\n \
      in vec4 fragnormal;\n \
      \n \
      layout (location = 0) out vec4 color; \n \
      void main(void) \n \
      { \n \
        vec3 V = normalize(-fragposition.xyz);  \n \
        vec3 N = normalize(fragnormal.xyz); \n \
        if (dot(N,V) < 0.0) { \n \
          N *= -1.0; \n \
        } \n \
        float intensity = pow(dot(N,V), 3.0); \n \
        color = vec4(fragcolor.xyz, 1.0) * clamp(intensity, 0.3, 1.0); \n \
        //color = vec4(fragnormal.xyz,1.0); \n \
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
    tml::beziersurface<tml::point<float,6> > surface;

    float d00[] = {0.0f, 0.2f, 1.0f, 0.0f, 0.0f, 0.0f};
    float d01[] = {0.4f, 1.0f, 2.4f, 0.5f, 0.0f, 0.0f};
    float d02[] = {0.2f, 2.4f, 0.3f, 1.0f, 0.0f, 0.0f};

    float d10[] = {1.2f, 0.0f, 0.0f, 0.0f, 0.3f, 0.0f};
    float d11[] = {1.4f, 1.4f, 0.5f, 0.5f, 0.3f, 0.0f};
    float d12[] = {1.0f, 2.0f, 0.2f, 1.0f, 0.3f, 0.0f};

    float d20[] = {3.1f, 0.3f, 1.0f, 0.0f, 0.6f, 0.0f};
    float d21[] = {3.5f, 1.2f, 0.4f, 0.5f, 0.6f, 0.0f};
    float d22[] = {3.0f, 2.0f, 0.4f, 1.0f, 0.6f, 0.0f};

    float d30[] = {4.5f, 0.2f, 1.0f, 0.0f, 1.0f, 0.0f};
    float d31[] = {3.4f, 1.5f, 0.7f, 0.5f, 1.0f, 0.0f};
    float d32[] = {4.0f, 2.2f, 0.1f, 1.0f, 1.0f, 0.0f};

    tml::point<float, 6> p00(d00);
    tml::point<float, 6> p01(d01);
    tml::point<float, 6> p02(d02);

    tml::point<float, 6> p10(d10);
    tml::point<float, 6> p11(d11);
    tml::point<float, 6> p12(d12);

    tml::point<float, 6> p20(d20);
    tml::point<float, 6> p21(d21);
    tml::point<float, 6> p22(d22);

    tml::point<float, 6> p30(d30);
    tml::point<float, 6> p31(d31);
    tml::point<float, 6> p32(d32);

    // first line
    surface.add(p00);
    surface.add(p01);
    surface.add(p02);

    surface.add(p10);
    surface.add(p11);
    surface.add(p12);

    surface.add(p20);
    surface.add(p21);
    surface.add(p22);

    surface.add(p30);
    surface.add(p31);
    surface.add(p32);

    surface.order_u(3);
    surface.order_v(4);

    std::vector<glpp::vec3f> attrib0; // x,y,z
    std::vector<glpp::vec3f> attrib1; // r,g,b
    std::vector<glpp::vec3f> attrib2; // normal
    
    float const domain_overlap = 1.0f;

    for (unsigned i = 0; i != _samples; ++i)
    {
      for (unsigned j = 0; j != _samples; ++j)
      {
        //float u = float(i) / (_samples-1);
        //float v = float(j) / (_samples-1);
        float u = -domain_overlap + (1.0f + 2.0f * domain_overlap ) * float(j) / (_samples-1);
        float v = -domain_overlap + (1.0f + 2.0f * domain_overlap ) * float(i) / (_samples-1);

        tml::point<float,6> pt, du, dv;
        
        //surface.evaluate(u, v, pt, du, dv);
        surface.evaluate(u, v, pt, du, dv, tml::horner<tml::point<float, 6>>());

        glpp::vec3f du_euclid(du[0], du[1], du[2]);
        glpp::vec3f dv_euclid(dv[0], dv[1], dv[2]);

        du_euclid.normalize();
        dv_euclid.normalize();

        glpp::vec3f n = cross(du_euclid, dv_euclid);

        attrib0.push_back(glpp::vec3f(pt[0], pt[1], pt[2]));
        attrib1.push_back(glpp::vec3f(pt[3], pt[4], pt[5]));
        attrib2.push_back(n);
      }
    }

    std::vector<unsigned> indices;

    for (unsigned i = 0; i != _samples-1; ++i)
    {
      for (unsigned j = 0; j != _samples-1; ++j)
      {
        int base = j + i * int(_samples); 

        int A = base;
        int B = base + 1;
        int C = base + 1 + int(_samples);
        int D = base + int(_samples);

        // two triangles for quad -> CCW
        indices.push_back(A); // 0
        indices.push_back(B); // 1
        indices.push_back(D); // 3
        
        indices.push_back(D); // 1
        indices.push_back(B); // 2
        indices.push_back(C); // 3  
      }
    }

    _indices.update(indices.begin(), indices.end());

    // reserve and then copy manually by mapping
    _attrib0.update(attrib0.begin(), attrib0.end());
    _attrib1.update(attrib1.begin(), attrib1.end());
    _attrib2.update(attrib2.begin(), attrib2.end());
    
    _vao.bind();

    _vao.attrib_array   (_attrib0, 0, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (0);

    _vao.attrib_array   (_attrib1, 1, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (1);

    _vao.attrib_array   (_attrib2, 2, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (2);

    _vao.unbind();
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

    _vao.bind();

    _indices.bind();

    _program.begin();

    _program.set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program.set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program.set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    //glPushAttrib(GL_BLEND);

    //glEnable    (GL_BLEND);
    //glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glDrawElements(GL_TRIANGLES, GLsizei((_samples-1)*(_samples-1)*2*3), GL_UNSIGNED_INT, 0);

    //glDrawArrays(GL_POINTS, 0, GLsizei(_samples*_samples));

    //glPopAttrib();

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
  glpp::elementarraybuffer            _indices;

  std::size_t                         _samples;
};


int main(int argc, char** argv)
{
  glewExperimental = true;

  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 4, 1, true);
  glewInit();

  application app;
  app.run();

  return 0;
}
