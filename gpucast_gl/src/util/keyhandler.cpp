/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : keyhandler.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/gl/util/keyhandler.hpp"

// header, system
#include <cassert> // assert

#include <cmath>

// header, project

namespace gpucast { namespace gl {

  ////////////////////////////////////////////////////////////////////////////////
  keyhandler::keyhandler()
    : _callbacks()
  {}



  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ keyhandler::~keyhandler( )
  {}


  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  keyhandler::mouse(enum button b, enum state s, int x, int y)
  {}


  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  keyhandler::motion(int x, int y)
  {}


  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  keyhandler::passivemotion ( int, int )
  {}

  ////////////////////////////////////////////////////////////////////////////////
  void 
  keyhandler::register_callback ( std::function<void(unsigned char, int, int)> const& callback )
  {
    _callbacks.push_back(callback);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  keyhandler::keyboard ( unsigned char b, int x, int y)
  {
    for (auto i = _callbacks.begin(); i != _callbacks.end(); ++i )
    {
      (*i)(b, x, y);
    }
  }

  } } // namespace gpucast / namespace gl
