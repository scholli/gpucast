/********************************************************************************
* 
* Copyright (C) 2009-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : contextinfo.hpp
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_CONTEXTINFO_HPP
#define GPUCAST_GL_CONTEXTINFO_HPP

#include <cstdlib>
#include <iostream>
#include <string>

#if WIN32
  #pragma warning(disable: 4245)
#endif

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

GPUCAST_GL void print_contextinfo  ( std::ostream& os );
GPUCAST_GL void print_extensions   ( std::ostream& os );
GPUCAST_GL void print_memory_usage ( std::ostream& os );
GPUCAST_GL bool check_extension    ( std::string const& extension_name );

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_HUTILS_HPP
