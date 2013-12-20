/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : parametric_volume.cpp
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

#include <tml/parametric/beziervolume.hpp>
#include <tml/parametric/beziersurface.hpp>
#include <tml/parametric/beziercurve.hpp>
#include <tml/parametric/point.hpp>
#include <glpp/util/timer.hpp>



class application
{
public :

  application()
    : _program  (),
      _camera   (),
      _trackball(new glpp::trackball),
      _vao      (),
      _indexbuffer_samples      (),
      _indexbuffer_derivatives  (),
      _indexbuffer_points       (),
      _attrib0  (),
      _attrib1  (),
      _attrib2  (),
      _samples  (16)
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
      layout (location = 1) in vec4 color;    \n \
      \n \
      uniform mat4 modelviewprojectionmatrix; \n \
      uniform mat4 modelviewmatrix; \n \
      uniform mat4 normalmatrix; \n \
      \n \
      out vec4 fragposition;\n \
      out vec4 fragcolor;\n \
      \n \
      void main(void) \n \
      { \n \
        fragposition = modelviewmatrix * vertex; \n \
        fragcolor    = color; \n \
        gl_Position  = modelviewprojectionmatrix * vertex; \n \
      }\n";

    std::string fragmentshader_code =
     "#version 330 compatibility\n \
      \n \
      in vec4 fragposition;\n \
      in vec4 fragcolor;\n \
      \n \
      layout (location = 0) out vec4 color; \n \
      void main(void) \n \
      { \n \
        float intensity = pow(length(fragcolor.xyz), 2.0); \n \
        color = vec4(fragcolor.xyz, intensity); \n \
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
     tml::beziercurve<tml::point<float,6> >   curve;

    float d00[] = {0.0f, 0.2f, 0.0f, 1.0f, 0.0f, 0.0f};
    float d01[] = {0.4f, 1.0f, 0.4f, 0.0f, 0.0f, 0.0f};
    float d02[] = {0.2f, 2.4f, 0.2f, 0.0f, 0.0f, 1.0f};

    float d10[] = {1.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float d11[] = {1.4f, 1.4f, 1.4f, 0.0f, 0.0f, 0.0f};
    float d12[] = {1.0f, 2.0f, 0.2f, 0.0f, 0.0f, 0.0f};

    float d20[] = {3.1f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f};
    float d21[] = {3.3f, 1.0f, 2.2f, 0.0f, 0.0f, 0.0f};
    float d22[] = {3.0f, 2.0f, 0.2f, 0.0f, 0.0f, 0.0f};

    float d30[] = {4.5f, 0.2f, 0.0f, 0.0f, 1.0f, 0.0f};
    float d31[] = {3.4f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f};
    float d32[] = {4.0f, 2.2f, 0.2f, 1.0f, 1.0f, 0.0f};

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

    float dc0[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float dc1[] = {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float dc2[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float dc3[] = {1.0f, 0.0f, 3.0f, 0.0f, 0.0f, 1.0f};
    float dc4[] = {0.0f, 1.0f, 5.0f, 0.0f, 0.0f, 1.0f};

    tml::point<float, 6> c0(dc0);
    tml::point<float, 6> c1(dc1);
    tml::point<float, 6> c2(dc2);
    tml::point<float, 6> c3(dc3);
    tml::point<float, 6> c4(dc4);

    curve.add(c0);
    curve.add(c1);
    curve.add(c2);
    curve.add(c3);
    curve.add(c4);

    tml::beziervolume<tml::point<float, 6> > volume = surface.extrude(curve);

    std::vector<glpp::vec3f> attrib0;
    std::vector<glpp::vec3f> attrib1;

    std::vector<int> indices_samples;
    std::vector<int> indices_derivatives;
    std::vector<int> indices_points;

    float const domain_overlap = 0.3f;
    float const derivative_scale = 0.1f;

    for (unsigned i = 0; i != _samples; ++i)
    {
      for (unsigned j = 0; j != _samples; ++j)
      {
        for (unsigned k = 0; k != _samples; ++k)
        {
          float u = -domain_overlap + (2.0f * domain_overlap + 1.0f) * float(i) / (_samples-1);
          float v = -domain_overlap + (2.0f * domain_overlap + 1.0f) * float(j) / (_samples-1);
          float w = -domain_overlap + (2.0f * domain_overlap + 1.0f) * float(k) / (_samples-1);

          tml::point<float,6> pt;
          tml::point<float,6> du, dv, dw;

          //volume.evaluate(u, v, w, pt, du, dv, dw);
          volume.evaluate(u, v, w, pt, du, dv, dw, tml::horner<tml::point<float,6> >() );

          du *= derivative_scale;
          dv *= derivative_scale;
          dw *= derivative_scale;

          indices_samples.push_back(int(attrib0.size()));
          attrib0.push_back(glpp::vec3f(pt[0], pt[1], pt[2]));

          tml::intervalf domain(0.0f, 1.0f);
          if ( !domain.in(u) || !domain.in(v) || !domain.in(w) )
          {
            attrib1.push_back(glpp::vec3f(1.0f, 1.0f, 1.0f));
          } else {
            attrib1.push_back(glpp::vec3f(pt[3], pt[4], pt[5]));
          }

          indices_derivatives.push_back(int(attrib0.size()));
          attrib0.push_back(glpp::vec3f(pt[0], pt[1], pt[2]));

          indices_derivatives.push_back(int(attrib0.size()));
          attrib0.push_back(glpp::vec3f(pt[0]+du[0], pt[1]+du[1], pt[2]+du[2]));
          attrib1.push_back(glpp::vec3f(1.0, 0.0, 0.0));
          attrib1.push_back(glpp::vec3f(1.0, 0.0, 0.0));

          indices_derivatives.push_back(int(attrib0.size()));
          attrib0.push_back(glpp::vec3f(pt[0], pt[1], pt[2]));

          indices_derivatives.push_back(int(attrib0.size()));
          attrib0.push_back(glpp::vec3f(pt[0]+dv[0], pt[1]+dv[1], pt[2]+dv[2]));
          attrib1.push_back(glpp::vec3f(0.0, 1.0, 0.0));
          attrib1.push_back(glpp::vec3f(0.0, 1.0, 0.0));

          indices_derivatives.push_back(int(attrib0.size()));
          attrib0.push_back(glpp::vec3f(pt[0], pt[1], pt[2]));

          indices_derivatives.push_back(int(attrib0.size()));
          attrib0.push_back(glpp::vec3f(pt[0]+dw[0], pt[1]+dw[1], pt[2]+dw[2]));
          attrib1.push_back(glpp::vec3f(0.0, 0.0, 1.0));
          attrib1.push_back(glpp::vec3f(0.0, 0.0, 1.0));
        }
      }
    }

    for ( tml::beziervolume<tml::point<float, 6> >::const_iterator cp = volume.begin(); cp != volume.end(); ++cp)
    {
      indices_points.push_back(int(attrib0.size()));
      attrib0.push_back(glpp::vec3f((*cp)[0], (*cp)[1], (*cp)[2]));
      attrib1.push_back(glpp::vec3f((*cp)[3], (*cp)[4], (*cp)[5]));
    }

    // copy data to buffer
    _attrib0.update(attrib0.begin(), attrib0.end());
    _attrib1.update(attrib1.begin(), attrib1.end());

    _indexbuffer_samples.update     (indices_samples.begin(),     indices_samples.end());
    _indexbuffer_derivatives.update (indices_derivatives.begin(), indices_derivatives.end());
    _indexbuffer_points.update      (indices_points.begin(),      indices_points.end());

    // bind vertex
    _vao.bind();
    _vao.attrib_array   (_attrib0, 0, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (0);
    _vao.attrib_array   (_attrib1, 1, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (1);
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

    _program.begin();

    _program.set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program.set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program.set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    _vao.bind();

    // draw samples
    glPointSize(2.0);
    _indexbuffer_samples.bind();
    glDrawElements(GL_POINTS, GLsizei(_samples*_samples*_samples), GL_UNSIGNED_INT, 0);
    _indexbuffer_samples.unbind();

    // draw control points
    glPointSize(5.0);
    _indexbuffer_points.bind();
    glDrawElements(GL_POINTS, GLsizei(3*4*5), GL_UNSIGNED_INT, 0);
    _indexbuffer_points.unbind();
    /*
    // draw partial derivatives
    _indexbuffer_derivatives.bind();
    //_indexbuffer_derivatives.set_index_pointer(GL_INT, 0, 0);
    glDrawElements(GL_LINES, 6 * GLsizei(_samples*_samples*_samples), GL_UNSIGNED_INT, 0);
    _indexbuffer_derivatives.unbind();
    */
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

  glpp::elementarraybuffer            _indexbuffer_samples;
  glpp::elementarraybuffer            _indexbuffer_derivatives;
  glpp::elementarraybuffer            _indexbuffer_points;

  glpp::arraybuffer                   _attrib0;
  glpp::arraybuffer                   _attrib1;
  glpp::arraybuffer                   _attrib2;

  std::size_t                         _samples;
};


int main(int argc, char** argv)
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 3, 3, false);
  glewInit();

  application app;
  app.run();

  return 0;
}
