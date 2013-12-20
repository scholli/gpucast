/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : stereo_anaglyph.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_STEREO_ANAGLYPH_HPP
#define GPUCAST_GL_STEREO_ANAGLYPH_HPP

// header, system
#include <string>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/camera.hpp>

#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/texture2d.hpp>

#include <gpucast/gl/math/vec3.hpp>


namespace gpucast { namespace gl {

  enum stereomode { active,
                    checkerboard,
                    passive,
                    anaglyph };

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL stereocamera : public camera
{
public :
  stereocamera                    ( float eyedistance = 0.1);

  virtual           ~stereocamera ( );
  virtual void      resize        ( std::size_t, std::size_t );
  virtual void      draw          ( );

  void              display       ( GLint drawbuffer,
                                    GLint viewport[4] );

  float             eyedistance   ( ) const;
  void              eyedistance   ( float );

  stereomode        mode          ( ) const;
  void              mode          ( stereomode );

private :

  void              _init_shader  ( );
  void              _init_anaglyph( );
  void              _init_checkerboard( );

  void              _init_fbo     ( );

  void              _init_texture ( GLint             texid,
                                    GLsizei           width,
                                    GLsizei           height );

public :

  float                             _eyedistance;

  framebufferobject           _fbo;
  renderbuffer                _depthbuffer;
  renderbuffer                _stencilbuffer;

  std::shared_ptr<program>  _anaglyph;
  std::shared_ptr<program>  _checkerboard;

  texture2d                   _left;
  texture2d                   _right;
};

} } // namespace gpucast / namespace gl

#endif //GPUCAST_GL_STEREO_ANAGLYPH_HPP
