/********************************************************************************
*
* Copyright (C) 2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : timer.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TRANSFORMATION_SEQUENCE_HPP
#define GPUCAST_GL_TRANSFORMATION_SEQUENCE_HPP

// header, system
#include <deque>

// header, project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/math/matrix4x4.hpp>

namespace gpucast { namespace gl {

struct GPUCAST_GL transformation_set
{
  gpucast::math::matrix4f modelview;
  gpucast::math::matrix4f modelview_inverse;
  gpucast::math::matrix4f modelviewprojection;
  gpucast::math::matrix4f modelviewprojection_inverse;
  gpucast::math::matrix4f normalmatrix;

  transformation_set () {}

  transformation_set ( gpucast::math::matrix4f const& mv, 
                       gpucast::math::matrix4f const& mvi,
                       gpucast::math::matrix4f const& mvp,
                       gpucast::math::matrix4f const& mvpi,
                       gpucast::math::matrix4f const& nm ) 
   : 
     modelview                    ( mv   ),
     modelview_inverse            ( mvi  ),
     modelviewprojection          ( mvp  ),
     modelviewprojection_inverse  ( mvpi ),
     normalmatrix                 ( nm   )
  {}

  ~transformation_set()
  {}

  void write ( std::ostream& os ) const
  {
    modelview.write(os);
    modelview_inverse.write(os);
    modelviewprojection.write(os);
    modelviewprojection_inverse.write(os);
    normalmatrix.write(os);
  }

  void read ( std::istream& is )
  {
    modelview.read(is);
    modelview_inverse.read(is);
    modelviewprojection.read(is);
    modelviewprojection_inverse.read(is);
    normalmatrix.read(is); 
  }

};

class GPUCAST_GL transformation_sequence
{
public :

  transformation_sequence  ();
  ~transformation_sequence ();

public :

  void                      add   ( gpucast::math::matrix4f const& mv,
                                    gpucast::math::matrix4f const& mvi, 
                                    gpucast::math::matrix4f const& mvp,
                                    gpucast::math::matrix4f const& mvpi,
                                    gpucast::math::matrix4f const& nm );

  void                      clear ();

  bool                      empty () const;

  transformation_set const& next  () const;

  void                      pop   ();

  void                      write ( std::string const& file ) const;
  void                      read  ( std::string const& file );

  void                      write ( std::ostream& os ) const;
  void                      read  ( std::istream& is );

private : // method

private : // members

  std::deque<transformation_set> _transformations;
};

} } // namespace gpucast / namespace gl

#endif // TIMER_HPP

