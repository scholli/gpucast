/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : fileloader.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>
#include <boost/optional.hpp>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/error.hpp>


#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <gpucast/gl/import/fileimport.hpp>

#include <gpucast/gl/graph/geode.hpp>
#include <gpucast/gl/graph/drawvisitor.hpp>


// visitor visists each node and applies the location
// set within the shader to the attribute arrays of each geode
class set_attribute_location_visitor : public gpucast::gl::visitor 
{
public :

  void accept ( gpucast::gl::geode& g ) const
  {
    for (unsigned i = 0; i < g.attributes(); ++i) 
    {
      switch (g.get_attribute_type(i)) 
      {
        case gpucast::gl::geode::vertex :
          g.set_attribute_location(i, 0); // shader expects vertices on location 0
          break;

        case gpucast::gl::geode::texcoord :
          g.set_attribute_location(i, 1); // shader expects texcoord on location 1
          break;

        case gpucast::gl::geode::normal : 
          g.set_attribute_location(i, 2); // shader expects normal on location 2
          break;

        default : 
          break; // ignore other attributes
      };
    }
  }

  void accept (gpucast::gl::group& grp) const
  {
    // groups have no ageometry, but visit all children
    std::for_each(grp.begin(), grp.end(), boost::bind(&gpucast::gl::node::visit, _1, *this));
  }

};


// draw phong visitor traverses all nodes and applies the 
// necessary material uniforms to shader
class draw_phong_visitor : public gpucast::gl::drawvisitor 
{
public :

  draw_phong_visitor( std::shared_ptr<gpucast::gl::program> p )
    : drawvisitor   ( ),
      _phongshader  (p)
  {}

  virtual void accept ( gpucast::gl::geode& g ) const
  {
    gpucast::gl::material m = g.get_material();

    _phongshader->set_uniform4fv("ka", 1, &m.ambient[0]);
    _phongshader->set_uniform4fv("kd", 1, &m.diffuse[0]);
    _phongshader->set_uniform4fv("ks", 1, &m.specular[0]);

    _phongshader->set_uniform1f("opacity", m.opacity);
    _phongshader->set_uniform1f("shininess", m.shininess);

    _phongshader->set_uniform4f("lightpos", 0.0f, 0.0f, 0.0f, 1.0f);

    g.draw();
  }

  private :

    std::shared_ptr<gpucast::gl::program> _phongshader;

};




class application
{
public :
 
  application()
    : _program  (new gpucast::gl::program),
      _root     (),
      _trackball(new gpucast::gl::trackball),
      _center   (),
      _scale    ()
  {
    // init phong shading program
    init_shader();
    init_gl();

    // add trackball interacion 
    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));
  }

  ~application() 
  {}

  void init_gl() const
  {
    glEnable(GL_DEPTH_TEST);
  }

  void load(std::string const& filename) 
  {
    // create fileimporter
    gpucast::gl::fileimport fi;  

    // load file
    _root = fi.load(filename);
    //_root = fi.load("../../mod_MOTORE.obj");

    // compute bounding box of imported object
    _root->compute_bbox();

    // use bounding information to compute center and scale
    _center = (_root->bbox().min + _root->bbox().max) / 2.0f;
    _scale  = (_root->bbox().max - _root->bbox().min).abs();

    // set the attribute bindings of all geodes
    _root->visit(set_attribute_location_visitor());
  }


  void init_shader()
  {
    // create vertex and fragment shader
    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);

    // load source code
    vs.load("./phong.vert");
    fs.load("./phong.frag");
  
    // compile shader
    vs.compile();
    fs.compile(); 

    // attach shader to program
    _program->add(&vs);
    _program->add(&fs);

    // finally link program
    _program->link();  
  }

  
  void draw()   
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f); 

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation() * 
                           gpucast::math::make_scale(5.0f/_scale, 5.0f/_scale, 5.0f/_scale) *
                           gpucast::math::make_translation(-_center[0], -_center[1], -_center[2]);
                          

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv   = view * model;
    gpucast::math::matrix4f mvp  = proj * mv;
    gpucast::math::matrix4f nm   = mv.normalmatrix();

    // use program to draw evereything
    _program->begin();
  
    _program->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program->set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program->set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    // draw_phong_visitor sets material dependent uniforms and triggers draw call of all geodes
    _root->visit(draw_phong_visitor(_program));

    _program->end();
  }
  

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


public :

  std::shared_ptr<gpucast::gl::program>    _program;
  std::shared_ptr<gpucast::gl::node>       _root;
  std::shared_ptr<gpucast::gl::trackball>  _trackball;
  gpucast::math::vec3f                         _center;
  float                               _scale;
};


int main(int argc, char* argv[])
{
  gpucast::gl::glutwindow::init(argc, argv, 1024, 1024, 0, 0);

  glewExperimental = true;
  glewInit();

  application app;

  if (argc >= 2) 
  {
    app.load(argv[1]);
    app.run();
  } else {
    app.load("./cow.obj");
    app.run();
  }

  return 0;
}
