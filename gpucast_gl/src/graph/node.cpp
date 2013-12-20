/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : node.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/graph/node.hpp"

// header, system

// header, project
#include <gpucast/gl/graph/drawvisitor.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
node::node()
: _bbox()
{}


///////////////////////////////////////////////////////////////////////////////
node::~node()
{}


///////////////////////////////////////////////////////////////////////////////
node::bbox_t const&    
node::bbox() const
{
  return _bbox;
}


///////////////////////////////////////////////////////////////////////////////
std::string const&    
node::name( ) const
{
  return _name;
}


///////////////////////////////////////////////////////////////////////////////
void   
node::name( std::string const& n )
{
  _name = n;
}





} } // namespace gpucast / namespace gl
