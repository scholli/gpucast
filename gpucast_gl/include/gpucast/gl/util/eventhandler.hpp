/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : eventhandler.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_EVENTHANDLER_HPP
#define GPUCAST_GL_EVENTHANDLER_HPP

// header, system

// header, project
#include <gpucast/gl/glpp.hpp>


namespace gpucast { namespace gl {

    // abstract base class for eventhandling
    class GPUCAST_GL eventhandler
    {
    public:

      enum button { left = 1, right = 2, middle = 4 };
      enum state  { press = 0, release = 1 };

    public:

      eventhandler();

      virtual ~eventhandler();

    public:

      virtual void mouse(enum button,
      enum state,
        int x,
        int y) = 0;

      virtual void motion(int x,
        int y) = 0;

      virtual void passivemotion(int x,
        int y) = 0;

      virtual void keyboard(unsigned char key,
        int           x,
        int           y) = 0;

    private:

    };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_EVENTHANDLER_HPP
