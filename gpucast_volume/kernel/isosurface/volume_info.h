#ifndef LIBGPUCAST_VOLUME_INFO_H
#define LIBGPUCAST_VOLUME_INFO_H

#include <math/conversion.h>

///////////////////////////////////////////////////////////////////////////////
struct volume_info
{
  __device__ volume_info()
    : volume_data_id      (0),
      volume_points_id    (0),
      volume_order        (uint3_t(0,0,0)),
      volume_bbox_size    (0.0f),
      attribute_data_id   (0),
      attribute_points_id (0),
      outer_facemask      (0)
  {}

  unsigned volume_data_id;
  unsigned volume_points_id;
  uint3    volume_order;
  float    volume_bbox_size;

  unsigned outer_facemask;

  unsigned attribute_data_id;
  unsigned attribute_points_id;
  
  /*
  __device__ void
  load_from_surface_id ( float4 const* surfacebuffer, float4 const* volumebuffer, unsigned surface_id )
  {
    float4 surface_info0      = surfacebuffer[surface_id    ];
    float4 surface_info1      = surfacebuffer[surface_id + 1];
    float4 surface_info2      = surfacebuffer[surface_id + 2];  
    
    surface_unique_id         = floatBitsToInt(surface_info0.x);
    volume_unique_id          = floatBitsToInt(surface_info0.y);
    volume_id                 = floatBitsToInt(surface_info0.z);
    attribute_id              = floatBitsToInt(surface_info0.w);
                  
    object_id                 = floatBitsToInt(surface_info1.x);
    surface_mesh_id           = floatBitsToInt(surface_info1.y);
    surface_is_outer          = floatBitsToInt(surface_info2.x) == 1;
    surface_order             = uint2_t ( floatBitsToInt(surface_info2.y), floatBitsToInt(surface_info2.z) );
    surface_type              = floatBitsToInt(surface_info2.w);
                         
    float4 volume_info0       = volumebuffer [ volume_id    ];  
    float4 volume_info1       = volumebuffer [ volume_id + 1];  

    volume_bbox_size          = volume_info0.w;
    volume_order              = uint3_t ( floatBitsToInt(volume_info1.x),
                                          floatBitsToInt(volume_info1.y),
                                          floatBitsToInt(volume_info1.z) );
  }*/

  __device__ inline void
  load_from_volume_id ( float4 const* volumedatabuffer, float4 const* attributedatabuffer, unsigned id )
  {
    volume_data_id      = id;

    float4 volume_info0 = volumedatabuffer[id    ]; 
    float4 volume_info1 = volumedatabuffer[id + 1];

    volume_points_id    = floatBitsToInt(volume_info0.x);
    attribute_data_id   = floatBitsToInt(volume_info0.z);

    attribute_points_id = floatBitsToInt(attributedatabuffer[attribute_data_id].z);

    volume_bbox_size    = volume_info0.w;

    volume_order        = uint3_t ( floatBitsToInt(volume_info1.x),
                                    floatBitsToInt(volume_info1.y),
                                    floatBitsToInt(volume_info1.z));

    outer_facemask      = floatBitsToInt(volume_info1.w);
  }

};

#endif // LIBGPUCAST_TARGET_FUNCTION_H