/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : stereo_active.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header, system
#include <string>

#include <boost/function.hpp>

#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/programobject.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/float3_t.hpp>

#include <gpucast/gl/glut/displaysetup.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
// active stereo setup
///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL stereo_active : public displaysetup 
{
public :

  stereo_active(unsigned width, unsigned height, float3_t const& , float3_t const& );

  virtual ~stereo_active();

  virtual void display();
};

} } // namespace gpucast / namespace gl
