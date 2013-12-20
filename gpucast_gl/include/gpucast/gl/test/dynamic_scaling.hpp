/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_scaling.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_DYNAMIC_SCALING_HPP
#define GPUCAST_GL_DYNAMIC_SCALING_HPP

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/test/dynamic_transform.hpp>
#include <gpucast/gl/math/vec3.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL dynamic_scaling : public dynamic_transform
  {
  public :

    dynamic_scaling                 ( unsigned      discrete_steps,
                                      vec3f const&  original_size, 
                                      vec3f const&  target_size );

    virtual ~dynamic_scaling        ();

  public :

    /* virtual */ matrix4f    current_transform ();

  private :

    vec3f _originalsize;
    vec3f _targetsize;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_DYNAMIC_SCALING_HPP
