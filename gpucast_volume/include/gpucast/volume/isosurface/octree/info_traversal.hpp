/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : info_traversal.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_INFO_TRAVERSAL_HPP
#define GPUCAST_INFO_TRAVERSAL_HPP

// header, system
#include <boost/noncopyable.hpp>
#include <map>

#include <gpucast/math/axis_aligned_boundingbox.hpp>

// header, project
#include <gpucast/volume/isosurface/octree/nodevisitor.hpp>
#include <gpucast/volume/isosurface/octree/node.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
class info_traversal : public nodevisitor, public boost::noncopyable
{
public :

  typedef std::map<std::string, nurbsvolume::attribute_volume_type::boundingbox_type> minmax_map;

public :

  info_traversal                               ();
  ~info_traversal                              ();
                                
  /* virtual */ void            visit          ( ocnode& ) const;
                                
  void                          print          ( std::ostream& ) const;
                                
private :

  struct _impl_t;
  _impl_t* _impl;

};

} // namespace gpucast

#endif // GPUCAST_INFO_TRAVERSAL_HPP
