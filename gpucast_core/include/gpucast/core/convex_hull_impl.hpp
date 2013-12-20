/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : convex_hull_impl.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_CONVEX_HULL_IMPL_HPP
#define GPUCAST_CORE_CONVEX_HULL_IMPL_HPP

#include <stdexcept>

extern "C" {
#include <stdio.h>
#include <stdlib.h>

#include <libqhull/libqhull.h>
#include <libqhull/mem.h>
#include <libqhull/qset.h>
#include <libqhull/geom.h>
#include <libqhull/merge.h>
#include <libqhull/poly.h>
#include <libqhull/io.h>
#include <libqhull/stat.h>
}

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
// template for building convex hull
////////////////////////////////////////////////////////////////////////////////
template<int N, typename opoint_t, typename point_oiter_t, typename tri_id_oiter_t>
void
convex_hull_compute(double*         points,
	                  std::size_t     npoints,
	                  point_oiter_t   chull,
	                  tri_id_oiter_t  tri_ids,
	                  std::size_t     tri_id_offset)
{
  qh_init_A(0, 0, stderr, 0, 0);
  qh_init_B(points, int(npoints), int(N), false);
  qh_initflags((char*)"qhull Pp QJ");
  qh_qhull();
  qh_check_output();
  qh_triangulate();

  // set by FORALLfacets:
  facetT *facet;
  // set by FORALLvertices, FOREACHvertex_:
  vertexT *vertex;
  // set by FOREACHvertex_:
  vertexT **vertexp;

  coordT *point, *pointtemp;

  FORALLpoints {
    opoint_t p;
    std::copy(point, point + N, &p[0]);
    *chull = p;
  }

  FORALLfacets {
    setT const* const tri = qh_facet3vertex(facet);
    FOREACHvertex_(tri) {
      *tri_ids = qh_pointid(vertex->point) + (int)tri_id_offset;
    }
  }

  qh_freeqhull(!qh_ALL);

  int curlong, totlong;
  qh_memfreeshort(&curlong, &totlong);

  if(curlong || totlong) throw std::runtime_error("qhull memory was not freed");
}

} // namespace gpucast

#endif // GPUCAST_CORE_CONVEX_HULL_IMPL_HPP