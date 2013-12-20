/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : stereo_karo.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_STEREO_KARO_HPP
#define GPUCAST_GL_STEREO_KARO_HPP

// header, system
#include <string>

#include <boost/function.hpp>

#include <gpucast/gl/glut/display/displaysetup.hpp>
#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/math/vec3.hpp>


namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL stereo_karo : public displaysetup 
{
public :
  stereo_karo                 ( unsigned          width, 
                                unsigned          height, 
                                vec3f const&   screen_pos, 
                                vec3f const&   view_pos,
                                float             screenwidth,
                                float             screenheight );
                                            
  ~stereo_karo                ( );
  
  void          setup         ( );
  virtual void  display       ( );
  virtual void  resize        ( unsigned          width, 
                                unsigned          height );
        
  void          render        ( );

  void          render_eye    ( GLenum colorbuffer, 
                                float eye_offset_x );
  
  void          initFBO       ( );
  void          initTexture   ( GLint             texid, 
                                int               width, 
                                int               height );

  float         eyedistance   ( ) const;
  void          eyedistance   ( float );

public :

  // eye distance in mm
  float                      eyedistance_;

  // screen width and height in mm
  float                      screenwidth_;
  float                      screenheight_;

  framebufferobject          fbo_;
  renderbuffer               rb_;
  renderbuffer               sb_;
  program*                   program_;

  GLuint                     left_eye_texid_;
  GLuint                     rght_eye_texid_;
};

} } // namespace gpucast / namespace gl

#endif //GPUCAST_GL_STEREO_KARO_HPP
