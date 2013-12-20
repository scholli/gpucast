/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : displaysetup.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header, system
#include <string>

#include <gpucast/gl/math/vec3.hpp>

#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/renderbuffer.hpp>


///////////////////////////////////////////////////////////////////////////////
// display setup base class
///////////////////////////////////////////////////////////////////////////////
class displaysetup
{
public :

  displaysetup(std::string const& name,
               unsigned width,
               unsigned height,
               float screeenwidth,
               float screenheight);

  virtual ~displaysetup();

  virtual void init() = 0;
  virtual void display() = 0;
  virtual void resize(unsigned width, unsigned height);
  virtual void camera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ);

  unsigned width() const;
  unsigned height() const;

  void increase_eyedistance();
  void decrease_eyedistance();

private :

  std::string name_;
  unsigned width_;
  unsigned height_;

protected :

  float              eyedistance_;
                     
  float              screenwidth_;
  float              screenheight_;

  gpucast::gl::vec3f position_screen_;
  gpucast::gl::vec3f position_camera_;
};

///////////////////////////////////////////////////////////////////////////////
// mono setup
///////////////////////////////////////////////////////////////////////////////
class displaysetup_mono : public displaysetup
{
public :

  displaysetup_mono(unsigned width, unsigned height);
  virtual ~displaysetup_mono();

  virtual void init();
  virtual void display();

};


///////////////////////////////////////////////////////////////////////////////
// stereo karo - for samsung 3D TV
///////////////////////////////////////////////////////////////////////////////
class displaysetup_stereo_karo : public displaysetup
{
public :
  // 1700mm ^= 1480.0 : 830.1 (16 : 9)
  displaysetup_stereo_karo(unsigned width, unsigned height);
  ~displaysetup_stereo_karo();

  virtual void init();
  virtual void display();
  virtual void resize(unsigned width, unsigned height);

  virtual void camera(float px, float py, float pz,
              float targetx, float targety, float targetz);

  void render();

  void initFBO();
  void initTexture(gpucast::gl::texture2d&, int width, int height);

public :

  gpucast::gl::framebufferobject fbo_;
  gpucast::gl::renderbuffer      rb_;
  gpucast::gl::renderbuffer      sb_;
  gpucast::gl::program*          program_;

  gpucast::gl::texture2d         left_;
  gpucast::gl::texture2d         right_;
};


