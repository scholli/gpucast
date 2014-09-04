/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : draw_traversal.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_DRAW_TRAVERSAL_HPP
#define GPUCAST_DRAW_TRAVERSAL_HPP

// header, system
#include <gpucast/math/matrix4x4.hpp>

// header, project
#include <gpucast/volume/isosurface/octree/nodevisitor.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
class draw_traversal : public nodevisitor
{
public :

  draw_traversal ( gpucast::math::matrix4x4<float> const& mvp );

  /* virtual */ void      visit          ( ocnode& ) const;

private :

  gpucast::math::matrix4x4<float> _mvp;

};

} // namespace gpucast

#endif // GPUCAST_DRAW_TRAVERSAL_HPP
