/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : drawvisitor.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_DRAWVISITOR_HPP
#define GPUCAST_GL_DRAWVISITOR_HPP

#include <set>

#include <boost/shared_ptr.hpp>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/graph/visitor.hpp>

namespace gpucast { namespace gl {

  // forward declarations
  class geode;
  class group;

  // abstract visitor base class
  class GPUCAST_GL drawvisitor : public visitor
  {
    public :

      drawvisitor                     ( );
      virtual ~drawvisitor            ( );

    public :

      virtual void accept         ( geode& ) const;
      virtual void accept         ( group& ) const;
      
  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_DRAWVISITOR_HPP
