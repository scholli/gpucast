#ifndef LIBGPUCAST_FIND_RAY_EXIT_H
#define LIBGPUCAST_FIND_RAY_EXIT_H

#include "fragmentlist/classify_intersection.h"
#include "math/conversion.h"

///////////////////////////////////////////////////////////////////////////////
__device__ 
inline bool 
find_ray_exit ( uint4*   indexlist, 
                float4*  surfacebuffer,
                unsigned fragment_index,
                raystate ray_state,
                unsigned surface_unique_id,
                unsigned current_volume_id,
                float    newton_epsilon,
                uint3*   correction,
                float*   exit_depth )
{
  unsigned next_entry = fragment_index;
  uint4 entry         = indexlist[next_entry]; 

  while ( next_entry != 0 )
  {
    unsigned   surface_id  = entry.z;
    unsigned   unique_id   = floatBitsToInt ( surfacebuffer[surface_id].x );
    float2 uv              = unpack_uv ( entry.y );
    bool   is_intersection = uv.x > -0.5f;
    float  depth           = intBitsToFloat ( int_t ( entry.w ) );
  
    bool found_potential_exit = classify_intersection ( unique_id,
                                                        is_intersection,
                                                        depth,
                                                        newton_epsilon,
                                                        ray_state );
    if ( found_potential_exit ) 
    {
      *exit_depth = depth;
      if ( (*correction).x == 0 ) // no correction necessary 
      {
        return true;
      } else {
        unsigned volume_id          = floatBitsToInt ( surfacebuffer[surface_id].z );
        unsigned adjacent_volume_id = floatBitsToInt ( surfacebuffer[surface_id + 1].z );

        if ( volume_id == (*correction).y || volume_id == (*correction).z ) { // continue with volume
          (*correction).x = volume_id;
          (*correction).y = floatBitsToInt ( surfacebuffer[surface_id].w );
          return true;
        }

        if ( adjacent_volume_id == (*correction).y || adjacent_volume_id == (*correction).z ) { // continue with adjacent volume
          (*correction).x = adjacent_volume_id;
          (*correction).y = floatBitsToInt ( surfacebuffer[surface_id + 1].w );
          return true;
        }
        
        // volume id's do not match
        (*correction).x = 0;
        return false;
      } 
    } 

    // go to next entry
    next_entry  = entry.x;
    entry       = indexlist[next_entry]; 
  }

  return false;
}

#endif

