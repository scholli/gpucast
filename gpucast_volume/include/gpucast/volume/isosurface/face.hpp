/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : face.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_FACE_HPP
#define GPUCAST_FACE_HPP

// header, system
#include <memory>

// header, project
#include <gpucast/math/oriented_boundingbox.hpp>

namespace gpucast {

struct face
{
    unsigned                  surface_id;
    gpucast::math::interval<float>      attribute_range;
    gpucast::math::obbox3f              obb;
    bool                      outer;
};

typedef std::shared_ptr<face> face_ptr;

} // namespace gpucast

#endif // GPUCAST_FACE_HPP
