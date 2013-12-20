/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : fileimport.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_FILEIMPORT_HPP
#define GPUCAST_GL_FILEIMPORT_HPP

#include <string>

#include <boost/shared_ptr.hpp>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/graph/node.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL fileimport 
  {
  public : 

    fileimport();
    virtual ~fileimport();

    std::shared_ptr<node>  load      ( std::string const& filename ) const;

  private : // methods

    std::shared_ptr<node>  _parse    ( std::string const& filename ) const;

  private : // attributes

    class impl_t;
    std::shared_ptr<impl_t> _impl;
  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_FILEIMPORT_HPP
