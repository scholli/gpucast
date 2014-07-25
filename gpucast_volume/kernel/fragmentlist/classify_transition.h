#ifndef LIBGPUCAST_CLASSIFY_TRANSITION_H
#define LIBGPUCAST_CLASSIFY_TRANSITION_H

#include "math/constants.h"
#include "math/raystate.h"

__device__
inline bool 
classify_transition ( unsigned         uniquesurfaceid, 
                      bool             found_new_intersection, 
                      bool             contains_iso_value, 
                      bool             neighbor_contains_iso_value,
                      bool             is_outer_nurbs, 
                      unsigned         volume_buffer_id,
                      unsigned         neighbor_volume_buffer_id,
                      bool*            neighbor_swap,
                      uint3*           correction,
                      raystate&        ray_state )
{
  *neighbor_swap    = false;

  if ( found_new_intersection )                 // hit volume at new intersection
  {
    if ( is_outer_nurbs )                       // hit outer nurbs surface
    { 
      if ( ray_state.volume == volume_buffer_id ) // and exits nurbs volume
      {
        ray_state.volume = MAX_UINT;
        return false;
      } else {                                  // and enters nurbs volume
        ray_state.volume = volume_buffer_id;
        return contains_iso_value;
      }
    } else {                                    // intersects inner surface
      if ( ray_state.volume == volume_buffer_id ) // go to neighbor volume
      {
        ray_state.volume   = neighbor_volume_buffer_id; 
        *neighbor_swap   = true;
        return neighbor_contains_iso_value;
      } else {
        if ( ray_state.volume == neighbor_volume_buffer_id ) // comes from neighbor volume
        {
          ray_state.volume = volume_buffer_id;
          return contains_iso_value;
        } else {                                // after some empty space skipping
          /*if ( contains_iso_value ) {
            (*ray_state).z = volume_buffer_id;  // continue with element
            return contains_iso_value;
          } 
          if ( neighbor_contains_iso_value ) {  // else continue with neighbor
            (*ray_state).z = neighbor_volume_buffer_id;  // continue with neighbor element
            return neighbor_contains_iso_value;
          }
          (*ray_state).z = volume_buffer_id;            // shouldn't happen
          */
          (*correction).x = 1;
          (*correction).y = volume_buffer_id;
          (*correction).z = neighbor_volume_buffer_id;
          return true;
        }
      }
    }
  } else {
    return false;                               // ignore misintersection
  }
}

#endif

