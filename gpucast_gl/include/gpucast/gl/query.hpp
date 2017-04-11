/********************************************************************************
*
* Copyright (C) 2016 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : query.hpp
*  project    : gpucast::gl
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

    void         reset();
    virtual void begin() const = 0;
    virtual void end() const = 0;
    virtual bool is_available() const;

  private : // member

    unsigned _id;
};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_QUERY_HPP
