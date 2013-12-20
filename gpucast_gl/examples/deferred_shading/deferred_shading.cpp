/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : deferred_shading.cpp
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
#include <glpp/texture2d.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/error.hpp>
#include <glpp/framebufferobject.hpp>
#include <glpp/renderbuffer.hpp>

#include <glpp/util/camera.hpp>
#include <glpp/util/timer.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/math/matrix4x4.hpp>

#include <glpp/primitives/plane.hpp>
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
          g.set_attribute_location(i, 1); // shader expects vertices on location 1
          break;

        case glpp::geode::normal : 
          g.set_attribute_location(i, 2); // shader expects vertices on location 2
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
class draw_to_buffers_visitor : public glpp::drawvisitor 
{
public :

  draw_to_buffers_visitor( boost::shared_ptr<glpp::program> p )
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

    g.draw();
  }

  private :

    boost::shared_ptr<glpp::program> _phongshader;
};






class application
{
public :
 
  application()
    : _first_pass     (new glpp::program),
      _second_pass    (new glpp::program),
      _root           (),
      _ambienttexture (new glpp::texture2d),
      _diffusetexture (new glpp::texture2d),
      _speculartexture(new glpp::texture2d),
      _normaltexture  (new glpp::texture2d),
      _depthbuffer    (new glpp::renderbuffer),
      _fbo            (new glpp::framebufferobject),
      _trackball      (new glpp::trackball),
      _camera         (),
      _center         (),
      _scale          (),
      _plane          (new glpp::plane(0, -1, 1)),
      _mode           (0),
      _timer          ()
  {
    // init phong shading program
    init_shader ();
    init_gl     ();
    init_fbo    ();

    // add trackball interacion 
    glpp::glutwindow::instance().add_eventhandler(_trackball);
    glpp::glutwindow::instance().add_keyevent(' ', boost::bind(boost::mem_fn(&application::toggle_mode), this, _1, _2));

    // set drawcallback on camera and attach camera to window
    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);
  }

  ~application() 
  {}

  void init_gl() const
  {
    glEnable(GL_DEPTH_TEST);

    glEnable(glpp::texture2d::target());
  }

  void init_color_texture (boost::shared_ptr<glpp::texture2d> colorattachment)
  {
    colorattachment->teximage(0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, 0);

    colorattachment->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    colorattachment->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //colorattachment->set_parameteri(GL_TEXTURE_WRAP_S, GL_REPEAT);
    //colorattachment->set_parameteri(GL_TEXTURE_WRAP_T, GL_REPEAT);
  }

  void toggle_mode (int /*x*/, int /*y*/)
  {
    if (++_mode > 4) _mode = 0;
    std::cout << _mode << std::endl;
  }


  void init_fbo() 
  {
    // allocate memory for buffers
    init_color_texture(_ambienttexture);
    init_color_texture(_diffusetexture);
    init_color_texture(_speculartexture);
    init_color_texture(_normaltexture);

    _depthbuffer->set(GL_DEPTH_COMPONENT32F_NV, 1024, 1024);

    _ambienttexture->bind();
    _diffusetexture->bind();

    _fbo->attach_texture(*_ambienttexture,  GL_COLOR_ATTACHMENT0_EXT);
    _fbo->attach_texture(*_diffusetexture,  GL_COLOR_ATTACHMENT1_EXT);
    _fbo->attach_texture(*_speculartexture, GL_COLOR_ATTACHMENT2_EXT);
    _fbo->attach_texture(*_normaltexture,   GL_COLOR_ATTACHMENT3_EXT);
    _fbo->attach_renderbuffer(*_depthbuffer,     GL_DEPTH_ATTACHMENT_EXT);

    _fbo->bind();
    _fbo->status();
    _fbo->unbind();
  }

  void load(std::string const& filename) 
  {
    // create fileimporter
    glpp::fileimport fi;  

    // load file
    _root = fi.load(filename);

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
    vs.load("../first_pass.vert");
    fs.load("../first_pass.frag");
  
    // compile shader
    vs.compile();
    fs.compile(); 

    std::cout << vs.log() << std::endl;
    std::cout << fs.log() << std::endl;

    // attach shader to program
    _first_pass->add(&vs);
    _first_pass->add(&fs);

    // finally link program
    _first_pass->link();  

    vs.load("../second_pass.vert");
    fs.load("../second_pass.frag");

    vs.compile();
    fs.compile();

    std::cout << vs.log() << std::endl;
    std::cout << fs.log() << std::endl;

    _second_pass->add(&vs);
    _second_pass->add(&fs);

    _second_pass->link();
  }

  void draw()
  {
    _timer.start();

    draw_to_fbo(); 

    glFinish();
    _timer.stop();
    std::cout << "draw time : " << _timer.result() << "\r" << std::endl;

    draw_from_texture();
  }
  

  void draw_to_fbo()   
  {
    _fbo->bind();

    GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, 
                         GL_COLOR_ATTACHMENT1_EXT,
                         GL_COLOR_ATTACHMENT2_EXT, 
                         GL_COLOR_ATTACHMENT3_EXT};

    glDrawBuffersARB(4, buffers);

    //_fbo->status();

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

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
    _first_pass->begin();
  
    _first_pass->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _first_pass->set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _first_pass->set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    // draw_phong_visitor sets material dependent uniforms and triggers draw call of all geodes
    _root->visit(draw_to_buffers_visitor(_first_pass));

    _first_pass->end();

    _fbo->unbind();
  }
  
  
  void draw_from_texture()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);
                          
    glpp::matrix4f proj = glpp::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 100.0f);
    glpp::matrix4f mv   = view;
    glpp::matrix4f mvp  = proj * mv;

    // use program to draw evereything
    _second_pass->begin();

    _second_pass->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _second_pass->set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);

    _second_pass->set_texture2d("ambient", *_ambienttexture,    0 );
    _second_pass->set_texture2d("diffuse", *_diffusetexture,    1 );
    _second_pass->set_texture2d("specular", *_speculartexture,  2 );
    _second_pass->set_texture2d("normal", *_normaltexture,      3 );

    _second_pass->set_uniform1i("mode", _mode);

    _second_pass->set_uniform4f("lightpos", 0.0f, 0.0f, 0.0f, 1.0f);

    // draw plane 
    _plane->draw();

    _second_pass->end();
  }


  void run() 
  {
    glpp::glutwindow::instance().run();
  }


public :

  boost::shared_ptr<glpp::program>    _first_pass;
  boost::shared_ptr<glpp::program>    _second_pass;
  boost::shared_ptr<glpp::node>       _root;

  boost::shared_ptr<glpp::texture2d>  _ambienttexture;
  boost::shared_ptr<glpp::texture2d>  _diffusetexture;
  boost::shared_ptr<glpp::texture2d>  _speculartexture;
  boost::shared_ptr<glpp::texture2d>  _normaltexture;
  boost::shared_ptr<glpp::renderbuffer> _depthbuffer;

  boost::shared_ptr<glpp::framebufferobject>  _fbo;

  glpp::camera                        _camera;
  boost::shared_ptr<glpp::trackball>  _trackball;
  glpp::vec3f                         _center;
  float                               _scale;
  boost::shared_ptr<glpp::plane>      _plane;

  unsigned                            _mode;

  glpp::timer                         _timer;
};


int main(int argc, char* argv[])
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 3, 3, false);
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

