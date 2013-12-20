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
#include <glpp/glut/window.hpp>

#include <glpp/program.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/error.hpp>

#include <glpp/util/camera.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/math/matrix4x4.hpp>

#include <glpp/import/fileimport.hpp>

#include <glpp/graph/geode.hpp>
#include <glpp/graph/drawvisitor.hpp>


// visitor visists each node and applies the location
// set within the shader to the attribute arrays of each geode
class set_attribute_location_visitor : public glpp::visitor 
{
public :

  void accept ( glpp::geode& g ) const
  {
    for (unsigned i = 0; i < g.attributes(); ++i) 
    {
      switch (g.get_attribute_type(i)) 
      {
        case glpp::geode::vertex :
          g.set_attribute_location(i, 0); // shader expects vertices on location 0
          break;

        case glpp::geode::texcoord :
          g.set_attribute_location(i, 1); // shader expects texcoord on location 1
          break;

        case glpp::geode::normal : 
          g.set_attribute_location(i, 2); // shader expects normal on location 2
          break;

        default : 
          break; // ignore other attributes
      };
    }
  }

  void accept (glpp::group& grp) const
  {
    // groups have no ageometry, but visit all children
    std::for_each(grp.begin(), grp.end(), boost::bind(&glpp::node::visit, _1, *this));
  }

};


// draw phong visitor traverses all nodes and applies the 
// necessary material uniforms to shader
class draw_phong_visitor : public glpp::drawvisitor 
{
public :

  draw_phong_visitor( boost::shared_ptr<glpp::program> p )
    : drawvisitor   ( ),
      _phongshader  (p)
  {}

  virtual void accept ( glpp::geode& g ) const
  {
    glpp::material m = g.get_material();

    _phongshader->set_uniform4fv("ka", 1, &m.ambient[0]);
    _phongshader->set_uniform4fv("kd", 1, &m.diffuse[0]);
    _phongshader->set_uniform4fv("ks", 1, &m.specular[0]);

    _phongshader->set_uniform1f("opacity", m.opacity);
    _phongshader->set_uniform1f("shininess", m.shininess);

    _phongshader->set_uniform4f("lightpos", 0.0f, 0.0f, 0.0f, 1.0f);

    g.draw();
  }

  private :

    boost::shared_ptr<glpp::program> _phongshader;

};




class application
{
public :
 
  application()
    : _program  (new glpp::program),
      _root     (),
      _trackball(new glpp::trackball),
      _camera   (),
      _center   (),
      _scale    ()
  {
    // init phong shading program
    init_shader();
    init_gl();

    // add trackball interacion 
    glpp::glutwindow::instance().add_eventhandler(_trackball);

    // set drawcallback on camera and attach camera to window
    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);
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
    glpp::fileimport fi;  

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
    glpp::vertexshader vs;
    glpp::fragmentshader fs; 

    // load source code
    vs.load("../phong.vert");
    fs.load("../phong.frag");
  
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

    glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f); 

    glpp::matrix4f model = glpp::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation() * 
                           glpp::make_scale(5.0f/_scale, 5.0f/_scale, 5.0f/_scale) *
                           glpp::make_translation(-_center[0], -_center[1], -_center[2]);
                          

    glpp::matrix4f proj = glpp::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    glpp::matrix4f mv   = view * model;
    glpp::matrix4f mvp  = proj * mv;
    glpp::matrix4f nm   = mv.normalmatrix();

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
    glpp::glutwindow::instance().run();
  }


public :

  boost::shared_ptr<glpp::program>    _program;
  boost::shared_ptr<glpp::node>       _root;
  glpp::camera                        _camera;
  boost::shared_ptr<glpp::trackball>  _trackball;
  glpp::vec3f                         _center;
  float                               _scale;
};


int main(int argc, char* argv[])
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0);
  glewInit();

  application app;

  if (argc >= 2) 
  {
    app.load(argv[1]);
    app.run();
  } else {
    app.load("../cow.obj");
    app.run();
  }

  return 0;
}
