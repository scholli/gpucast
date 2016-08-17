/********************************************************************************
*
* Copyright (C) 2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : contextinfo.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/gl/util/contextinfo.hpp"

// header, system
#include <iomanip> // setprecision
#include <sstream>

#include <boost/log/trivial.hpp>

namespace gpucast { namespace gl {


GPUCAST_GL void print_contextinfo ( std::ostream& os )
{
  char* gl_version   = (char*)glGetString(GL_VERSION);
  char* glsl_version = (char*)glGetString(GL_SHADING_LANGUAGE_VERSION);

  GLint context_profile;
  glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &context_profile);

  BOOST_LOG_TRIVIAL(info) << "OpenGL Version String : " << gl_version << std::endl;
  BOOST_LOG_TRIVIAL(info) << "GLSL Version String   : " << glsl_version << std::endl;

  switch (context_profile) {
    case GL_CONTEXT_CORE_PROFILE_BIT :
      BOOST_LOG_TRIVIAL(info) << "Core Profile" << std::endl; break;
    case GL_CONTEXT_COMPATIBILITY_PROFILE_BIT :
      BOOST_LOG_TRIVIAL(info) << "Compatibility Profile" << std::endl; break;
    default :
      BOOST_LOG_TRIVIAL(info) << "Unknown Profile" << std::endl;
  };
}

GPUCAST_GL void print_extensions   ( std::ostream& os )
{
  int n_extensions = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &n_extensions);

  for ( int i = 0; i != n_extensions; ++i )
  {
    os << glGetStringi ( GL_EXTENSIONS, i ) << std::endl;
  }
}

GPUCAST_GL void print_memory_usage ( std::ostream& os )
{
  if ( check_extension("GL_NVX_gpu_memory_info") )
  {
    os << "renderer::dump_memory_info(): GL_NVX_gpu_memory_info unsupported), ignoring call." << std::endl;
  } else {
    int dedicated_vidmem         = 0;    
    int total_available_memory   = 0;
    int current_available_vidmem = 0;
    int eviction_count           = 0;
    int evicted_memory           = 0;
 
    glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX        , &dedicated_vidmem        );
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX  , &total_available_memory  );
    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &current_available_vidmem);
    glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX          , &eviction_count          );
    glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX          , &evicted_memory          );
     
    os << std::fixed << std::setprecision(3)
        << "dedicated_vidmem        : " << static_cast<float>(dedicated_vidmem        ) / 1024.0f << "MiB" << std::endl
        << "total_available_memory  : " << static_cast<float>(total_available_memory  ) / 1024.0f << "MiB" << std::endl
        << "current_available_vidmem: " << static_cast<float>(current_available_vidmem) / 1024.0f << "MiB" << std::endl
        << "eviction_count          : " << eviction_count           << std::endl
        << "evicted_memory          : " << evicted_memory           << std::endl;
  }
}

GPUCAST_GL bool check_extension ( std::string const& extension_name )
{
  std::stringstream sstr;
  print_contextinfo ( sstr );
  while ( sstr )
  {
    std::string extension;
    sstr >> extension;
    if ( extension == extension_name ) 
    {
      return GL_TRUE;
    }
  }

  return GL_FALSE;
}

} } // namespace gpucast / namespace gl

