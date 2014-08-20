/********************************************************************************
* 
* Copyright (C) 2011 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : vsync.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_VSYNC_HPP
#define GPUCAST_GL_VSYNC_HPP

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

GPUCAST_GL bool set_vsync ( bool enable );
GPUCAST_GL bool get_vsync ( bool& vsync );

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_VSYNC_HPP
