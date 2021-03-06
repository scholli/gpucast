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
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/shader.hpp>

#include <gpucast/gl/error.hpp>
#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/renderbuffer.hpp>


#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <gpucast/gl/primitives/plane.hpp>
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
          g.set_attribute_location(i, 1); // shader expects vertices on location 1
          break;

        case gpucast::gl::geode::normal : 
          g.set_attribute_location(i, 2); // shader expects vertices on location 2
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
class draw_to_buffers_visitor : public gpucast::gl::drawvisitor 
{
public :

  draw_to_buffers_visitor( std::shared_ptr<gpucast::gl::program> p )
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

    g.draw();
  }

  private :

    std::shared_ptr<gpucast::gl::program> _phongshader;
};






class application
{
public :
 
  application()
    : _first_pass     (new gpucast::gl::program),
      _second_pass    (new gpucast::gl::program),
      _root           (),
      _ambienttexture (new gpucast::gl::texture2d),
      _diffusetexture (new gpucast::gl::texture2d),
      _speculartexture(new gpucast::gl::texture2d),
      _normaltexture  (new gpucast::gl::texture2d),
      _depthbuffer    (new gpucast::gl::renderbuffer),
      _fbo            (new gpucast::gl::framebufferobject),
      _trackball      (new gpucast::gl::trackball),
      _center         (),
      _scale          (),
      _plane          (new gpucast::gl::plane(0, -1, 1)),
      _mode           (0),
      _timer          ()
  {
    // init phong shading program
    init_shader ();
    init_gl     ();
    init_fbo    ();

    // add trackball interacion 
    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);
    gpucast::gl::glutwindow::instance().add_keyevent(' ', boost::bind(boost::mem_fn(&application::toggle_mode), this, _1, _2));

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));
  }

  ~application() 
  {}

  void init_gl() const
  {
    glEnable(GL_DEPTH_TEST);

    glEnable(gpucast::gl::texture2d::target());
  }

  void init_color_texture (std::shared_ptr<gpucast::gl::texture2d> colorattachment)
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
    gpucast::gl::fileimport fi;  

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
    gpucast::gl::shader   vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);

    // load source code
    vs.load("./first_pass.vert");
    fs.load("./first_pass.frag");
  
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

    vs.load("./second_pass.vert");
    fs.load("./second_pass.frag");

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

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);
                          
    gpucast::math::matrix4f proj = gpucast::math::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 100.0f);
    gpucast::math::matrix4f mv   = view;
    gpucast::math::matrix4f mvp  = proj * mv;

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
    gpucast::gl::glutwindow::instance().run();
  }


public :

  std::shared_ptr<gpucast::gl::program>    _first_pass;
  std::shared_ptr<gpucast::gl::program>    _second_pass;
  std::shared_ptr<gpucast::gl::node>       _root;

  std::shared_ptr<gpucast::gl::texture2d>  _ambienttexture;
  std::shared_ptr<gpucast::gl::texture2d>  _diffusetexture;
  std::shared_ptr<gpucast::gl::texture2d>  _speculartexture;
  std::shared_ptr<gpucast::gl::texture2d>  _normaltexture;
  std::shared_ptr<gpucast::gl::renderbuffer> _depthbuffer;

  std::shared_ptr<gpucast::gl::framebufferobject>  _fbo;

  std::shared_ptr<gpucast::gl::trackball>  _trackball;
  gpucast::math::vec3f                         _center;
  float                               _scale;
  std::shared_ptr<gpucast::gl::plane>      _plane;

  unsigned                            _mode;

  gpucast::gl::timer                         _timer;
};


int main(int argc, char* argv[])
{
  gpucast::gl::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 0, 0, true);
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

