/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : keyhandler.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_GL_KEYHANDLER_HPP
#define LIBGPUCAST_GL_KEYHANDLER_HPP

// header, system
#include <vector>

// header, project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/eventhandler.hpp>

#include <boost/function.hpp>

namespace gpucast { namespace gl {

class GPUCAST_GL keyhandler : public eventhandler
{
public :

public :

  keyhandler                         ( );

  /* virtual */ ~keyhandler          ( );

public :

  /* virtual */ void mouse          ( enum button, enum state, int x, int y );
  /* virtual */ void motion         ( int x, int y );
  /* virtual */ void passivemotion  ( int x, int y );
  /* virtual */ void keyboard       ( unsigned char, int, int);

  void register_callback ( std::function<void(unsigned char, int, int)> const& callback );

private :

  std::vector<std::function<void(unsigned char, int, int)>> _callbacks;

};

} } // namespace gpucast / namespace gl

#endif // LIBGPUCAST_GL_KEYHANDLER_HPP
