/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : error.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_ERROR_HPP
#define GPUCAST_GL_ERROR_HPP

#include <string>

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

  GPUCAST_GL extern bool error(std::string const& message);

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_GLSTATE_HPP
