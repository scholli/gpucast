/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : sampler.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_SAMPLER_HPP
#define GPUCAST_GL_SAMPLER_HPP

#if WIN32
  #pragma warning(disable: 4996) // unsafe std::copy
#endif

// header system
#include <boost/noncopyable.hpp>

// header project
#include <gpucast/gl/glpp.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL sampler : boost::noncopyable
{
public : // c'tor, d'tor

  sampler                         ();
  virtual ~sampler                ();

public : // methods

  void            swap            ( sampler& );

  void            parameter       ( GLenum pname,
                                    GLint  param ) const;

  void            parameter       ( GLenum  pname,
                                    GLfloat param ) const;

  void            bind            ( GLint unit ) const;
  void            unbind          () const;

private : // members

  GLuint          _id;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_SAMPLER_HPP
