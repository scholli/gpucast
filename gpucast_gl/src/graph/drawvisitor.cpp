/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : drawvisitor.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/graph/drawvisitor.hpp"

// header, system
#include <boost/bind.hpp>
#include <boost/mem_fn.hpp>

#include <gpucast/gl/graph/geode.hpp>
#include <gpucast/gl/graph/group.hpp>

// header, project

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
drawvisitor::drawvisitor()
  : visitor()
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ drawvisitor::~drawvisitor()
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void 
drawvisitor::accept ( geode& g) const
{
  g.draw();
}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void 
drawvisitor::accept ( group& g ) const
{
  std::for_each(g.begin(), g.end(), std::bind(&node::visit, std::placeholders::_1, std::ref(*this)));
}


} } // namespace gpucast / namespace gl

