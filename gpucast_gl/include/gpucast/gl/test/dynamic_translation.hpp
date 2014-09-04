/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_translation.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_DYNAMIC_TRANSLATION_HPP
#define GPUCAST_GL_DYNAMIC_TRANSLATION_HPP

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/test/dynamic_transform.hpp>
#include <gpucast/math/vec3.hpp>


namespace gpucast { namespace gl {

  class GPUCAST_GL dynamic_translation : public dynamic_transform
  {
  public :

    dynamic_translation         ( unsigned     discrete_steps,
                                  gpucast::math::vec3f const& source_position,
                                  gpucast::math::vec3f const& target_position );

    ~dynamic_translation        ();

  public :

    /* virtual */ gpucast::math::matrix4f current_transform ();

  private :

    gpucast::math::vec3f                   _source;
    gpucast::math::vec3f                   _target;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_DYNAMIC_TRANSLATION_HPP
