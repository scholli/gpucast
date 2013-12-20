/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : eventprinter.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_EVENTPRINTER_HPP
#define GPUCAST_GL_EVENTPRINTER_HPP

// header, system
#include <iostream>

// header, project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/eventhandler.hpp>

#include <boost/noncopyable.hpp>

namespace gpucast { namespace gl {

  // prints events on ostream
  class GPUCAST_GL eventprinter : public eventhandler, boost::noncopyable
  {
  public :

    eventprinter            ( std::ostream& );

    virtual ~eventprinter   ( );

  public :

    /* virtual */ void  mouse         ( int button,
                                        int state,
                                        int x,
                                        int y );

    /* virtual */ void  motion        ( int x,
                                        int y );

    /* virtual */ void  passivemotion ( int x,
                                        int y );

    /* virtual */ void  keyboard      ( unsigned char key,
                                        int           x,
                                        int           y );

  private :

    std::ostream& _os;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_EVENTPRINTER_HPP
