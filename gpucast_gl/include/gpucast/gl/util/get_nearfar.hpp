/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : get_nearfar.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_GET_NEARFAR_HPP
#define GPUCAST_GL_GLSTATE_HPP

// header, system
#include <string> // std::string

// header, project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/math/vec3.hpp>


namespace gpucast { namespace gl {

// get near far plane from state
GPUCAST_GL void get_nearfar(float& nearplane, float& farplane);

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_GLSTATE_HPP
