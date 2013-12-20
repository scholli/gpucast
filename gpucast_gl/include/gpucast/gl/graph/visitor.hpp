/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : visitor.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_VISITOR_HPP
#define GPUCAST_GL_VISITOR_HPP

#include <set>

#include <boost/shared_ptr.hpp>

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

  // forward declarations
  class geode;
  class group;

  // abstract visitor base class
  class GPUCAST_GL visitor
  {
    public :

      visitor                     ( );
      virtual ~visitor            ( );

    public :

      virtual void accept         ( geode& ) const =0;
      virtual void accept         ( group& ) const =0;
      
  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_NODEVISITOR_HPP
