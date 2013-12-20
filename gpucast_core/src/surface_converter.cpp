/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : surface_converter.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/surface_converter.hpp"

// header, system
#include <list>
#include <cmath>
#include <functional>
#include <thread>
#include <mutex>

// header, project
#include <gpucast/math/parametric/pointmesh2d.hpp>
#include <gpucast/math/parametric/algorithm/converter.hpp>

#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>
#include <gpucast/core/beziersurface.hpp>
#include <gpucast/core/nurbssurface.hpp>
#include <gpucast/core/trimdomain.hpp>


namespace gpucast {


class surface_converter::impl_t
{
public :
  impl_t()
    : target        ( ),
      source        ( ),
      next          ( ),
      source_access ( ),
      target_access ( ),
      nthreads      (8)
  {}

  std::shared_ptr<beziersurfaceobject>   target;
  std::shared_ptr<nurbssurfaceobject>    source;
  nurbssurfaceobject::const_iterator     next;
  std::mutex                             source_access;
  std::mutex                             target_access;
  std::size_t                            nthreads;
};


////////////////////////////////////////////////////////////////////////////////
surface_converter::surface_converter()
: _impl  (new impl_t)
{}

////////////////////////////////////////////////////////////////////////////////
surface_converter::~surface_converter()
{
  delete _impl;
}


////////////////////////////////////////////////////////////////////////////////
void
surface_converter::convert(std::shared_ptr<nurbssurfaceobject> const& ns, std::shared_ptr<beziersurfaceobject> const& bs)
{
  _impl->target = bs;
  _impl->target->clear();

  _impl->source = ns;
  _impl->next   = _impl->source->begin();

#if 1
  std::list<std::shared_ptr<std::thread> > threadpool;

  for (std::size_t i = 0; i != _impl->nthreads; ++i)
  {
    threadpool.push_back(std::shared_ptr<std::thread>(new std::thread(std::bind(&surface_converter::_fetch_task, this))));
  }

  std::for_each(threadpool.begin(), threadpool.end(), std::bind(&std::thread::join, std::placeholders::_1));
#else
  _fetch_task();
#endif

}


////////////////////////////////////////////////////////////////////////////////
void
surface_converter::_convert(nurbssurfaceobject::surface_type const& nurbspatch)
{
   gpucast::math::converter2d conv2d;
   gpucast::math::converter3d conv3d;

  // first convert trimmed nurbs surface into bezierpatches
  std::vector< gpucast::math::beziersurface_from_nurbs< gpucast::math::point3d> > bezierpatches;
  conv3d.convert(nurbspatch, std::back_inserter(bezierpatches));

  // then convert all trimming nurbs curves into bezier curves
  std::vector<trimdomain::contour_type>       loops;

  for ( nurbssurface::const_trimloop_iterator l = nurbspatch.trimloops().begin(); l != nurbspatch.trimloops().end(); ++l)
  {
    // first split nurbs loop into bezier loop
    beziersurface::curve_container     loop;
    for ( nurbssurface::const_curve_iterator c = l->begin(); c != l->end(); ++c )
    {
      conv2d.convert(*c, std::back_inserter(loop));
    }
 
    // convert to pointer
    std::vector<trimdomain::curve_ptr> loop_as_ptr;
    for ( auto c = loop.begin(); c != loop.end(); ++c )
    {
      loop_as_ptr.push_back ( trimdomain::curve_ptr ( new trimdomain::curve_type ( *c ) ) );
    }

    loops.push_back ( trimdomain::contour_type ( loop_as_ptr.begin(), loop_as_ptr.end() ) );
  }

  beziersurface::trimdomain_ptr domain ( new trimdomain );

  // add curves, set trimtype and which part of the nurbs patch the patch is
  std::for_each(loops.begin(), loops.end(), std::bind(&trimdomain::add, domain, std::placeholders::_1));
  domain->type         ( nurbspatch.trimtype());
  domain->nurbsdomain  ( trimdomain::bbox_type ( trimdomain::point_type ( nurbspatch.umin(), nurbspatch.vmin() ), 
                                                 trimdomain::point_type ( nurbspatch.umax(), nurbspatch.vmax() ) ) );

  // generate trimmed bezier patches from subpatches and trimming curves
  for (std::vector< gpucast::math::beziersurface_from_nurbs< gpucast::math::point3d> >::const_iterator b = bezierpatches.begin(); b != bezierpatches.end(); ++b)
  {
    std::shared_ptr<beziersurface> tbs(new beziersurface(b->surface));

    tbs->domain(domain);

    tbs->bezierdomain ( trimdomain::bbox_type ( trimdomain::point_type ( b->umin, b->vmin ), 
                                                trimdomain::point_type ( b->umax, b->vmax ) ) );

    // lock for write access into beziersurfaceobject
    std::lock_guard<std::mutex> lock(_impl->target_access);
    _impl->target->add(tbs);
  }
}


////////////////////////////////////////////////////////////////////////////////
void
surface_converter::_fetch_task ()
{
  bool something_to_do = true;

  while ( something_to_do )
  {
    // try to get a task
    std::lock_guard<std::mutex> lck(_impl->source_access);

    if ( _impl->next != _impl->source->end() )
    {
      nurbssurface const& task = *(_impl->next);
      ++(_impl->next);
      _convert( task );

    } else {
      something_to_do = false;
    }
  }

}

} // namespace gpucast
