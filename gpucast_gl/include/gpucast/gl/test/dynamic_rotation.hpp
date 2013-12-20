/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_rotation.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_DYNAMIC_ROTATION_HPP
#define GPUCAST_GL_DYNAMIC_ROTATION_HPP

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/test/dynamic_transform.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL dynamic_rotation : public dynamic_transform
  {
  public :

    dynamic_rotation              ( unsigned discrete_steps,
                                    unsigned rotation_axis );

    ~dynamic_rotation             ();

  public :

    /* virtual */ matrix4f  current_transform  ();

  private :

    unsigned                _rotation_axis;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_DYNAMIC_ROTATION_HPP
