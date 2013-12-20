/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : eventprinter.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/util/eventprinter.hpp"


namespace gpucast { namespace gl {

  /////////////////////////////////////////////////////////////////////////////
  eventprinter::eventprinter(std::ostream& os)
    : eventhandler(),
      _os(os)
  {}

  /////////////////////////////////////////////////////////////////////////////
  eventprinter::~eventprinter()
  {}

  /* virtual */ void
  eventprinter::mouse ( int button, int state, int x, int y )
  {
    _os << "mouse event :"        << std::endl
        << "  button: " << button << std::endl
        << "  state : " << state  << std::endl
        << "  x     : " << x      << std::endl
        << "  y     : " << y      << std::endl;
  }

  /* virtual */ void
  eventprinter::motion( int x, int y )
  {
    _os << "mouse motion :"   << std::endl
        << "  x     : " << x  << std::endl
        << "  y     : " << y  << std::endl;
  }

  /* virtual */ void
  eventprinter::passivemotion( int x, int y )
  {
    _os << "passive mouse motion :"   << std::endl
        << "  x     : " << x  << std::endl
        << "  y     : " << y  << std::endl;
  }

  /* virtual */ void
  eventprinter::keyboard ( unsigned char key, int x, int y)
  {
    _os << "keyboard event :"  << std::endl
        << "  key : " << key   << std::endl
        << "  x   : " << x     << std::endl
        << "  y   : " << y     << std::endl;
  }

} } // namespace gpucast / namespace gl
