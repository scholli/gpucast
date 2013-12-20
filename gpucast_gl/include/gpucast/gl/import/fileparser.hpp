/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : fileparser.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_FILEPARSER_HPP
#define GPUCAST_GL_FILEPARSER_HPP

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

  class node;
    
  class GPUCAST_GL fileparser
  {
  public : 

    fileparser();
    virtual ~fileparser();

  public : 

    virtual std::shared_ptr<node>   parse ( std::string const& filename ) =0;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_FILEIMPORT_HPP
