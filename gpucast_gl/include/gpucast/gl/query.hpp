/********************************************************************************
*
* Copyright (C) 1009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : query.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_QUERY_HPP
#define GPUCAST_GL_QUERY_HPP

// header system
#include <string>
#include <map>

// header project
#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL query
  {
  public :

    query();
    ~query();

  public : // methods

    unsigned id() const;

    virtual void begin() const = 0;
    virtual void end() const = 0;
    virtual bool is_available() const = 0;

  private : // member

    unsigned _id;
};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_QUERY_HPP
