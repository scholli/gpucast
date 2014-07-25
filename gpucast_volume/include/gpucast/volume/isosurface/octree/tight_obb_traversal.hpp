/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : tight_obb_traversal.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_TIGHT_OBB_TRAVERSAL_HPP
#define GPUCAST_TIGHT_OBB_TRAVERSAL_HPP

// header, system
#include <boost/noncopyable.hpp>
#include <gpucast/math/oriented_boundingbox.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/isosurface/octree/node.hpp>
#include <gpucast/volume/isosurface/octree/nodevisitor.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
class tight_obb_traversal : public nodevisitor, public boost::noncopyable
{
public :

  tight_obb_traversal();
  ~tight_obb_traversal();

  /* virtual */ void          visit          ( ocnode& ) const;

  gpucast::math::oriented_boundingbox<node::point_type> bbox           ();

private :

  std::list<node::point_type>* _points;

};

} // namespace gpucast

#endif // GPUCAST_TIGHT_OBB_TRAVERSAL_HPP
