/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nodevisitor.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_NODEVISITOR_HPP
#define GPUCAST_NODEVISITOR_HPP

// header, system
#include <memory>

// header, project
#include <gpucast/volume/gpucast.hpp>

namespace gpucast {

// forward declarations
class node;
class ocnode;

// visitor base class /////////////////////////////////////////////////////////
class GPUCAST_VOLUME nodevisitor
{
public : 

  typedef std::shared_ptr<node>     node_ptr;

public :

  nodevisitor();
  virtual ~nodevisitor();

public :

  virtual void      visit           ( ocnode& ) const = 0;
};

} // namespace gpucast

#endif // GPUCAST_NODEVISITOR_HPP
