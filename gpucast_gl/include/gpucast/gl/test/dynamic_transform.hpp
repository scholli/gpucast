/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_transform.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_DYNAMIC_TRANSFORM_HPP
#define GPUCAST_GL_DYNAMIC_TRANSFORM_HPP

#include <gpucast/gl/glpp.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <boost/shared_ptr.hpp>

namespace gpucast { namespace gl {

  class dynamic_transform 
  {
  public :

    dynamic_transform             ( unsigned discrete_steps );
    virtual ~dynamic_transform    ();

  public :

    bool                finished  () const;
    void                reset     ();
    void                step      ();
    
    virtual gpucast::math::matrix4f    current_transform   () = 0;

  protected :

    unsigned            _discrete_steps;  // max number of discrete steps for transformation
    unsigned            _current_step;    // number of current step in sequence

  };

  typedef std::shared_ptr<dynamic_transform>  dynamic_transform_ptr;

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_DYNAMIC_TRANSFORM_HPP
