/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : init_glew.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_INIT_GLEW_HPP
#define GPUCAST_GL_INIT_GLEW_HPP

#include <iostream>

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

GPUCAST_GL void init_glew ( std::ostream& log_os = std::cout );

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_INIT_GLEW_HPP
