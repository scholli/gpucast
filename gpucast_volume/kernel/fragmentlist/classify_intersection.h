#ifndef LIBGPUCAST_CLASSIFY_INTERSECTION_H
#define LIBGPUCAST_CLASSIFY_INTERSECTION_H

#include "math/raystate.h"

__device__ 
inline bool 
classify_intersection ( unsigned     unique_id,
                        bool         intersects_surface,
                        float        intersection_depth,
                        float        newton_epsilon,
                        raystate&    ray_state )
{
  bool has_same_depth          = fabs(intersection_depth - ray_state.depth) < newton_epsilon;
  bool has_same_unique_id      = ray_state.surface == unique_id;
  bool equals_last_transition  = has_same_unique_id && has_same_depth;

  return !equals_last_transition && intersects_surface;
}


#endif

