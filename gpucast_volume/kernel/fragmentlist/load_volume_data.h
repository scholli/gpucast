#ifndef LIBGPUCAST_CLASSIFY_LOAD_VOLUME_DATA_H
#define LIBGPUCAST_CLASSIFY_LOAD_VOLUME_DATA_H

__device__
inline void load_volume_data ( uint4 const*     surfacedatabuffer,
                               float4 const*    volumedatabuffer,
                               float4 const*    attributedatabuffer,
                               unsigned         surface_data_id,
                               uint2*           surface_order,
                               unsigned*        surface_type,
                               unsigned*        surface_mesh_id,
                               bool*            surface_is_outer,
                               unsigned*        volume_data_id,
                               unsigned*        volume_points_id,
                               unsigned*        attribute_data_id,
                               unsigned*        attribute_points_id,
                               uint3*           volume_order,
                               float*           volume_bbox_size,
                               unsigned*        adjacent_volume_data_id,
                               unsigned*        adjacent_attribute_id,
                               float*           adjacent_bbox_size,
                               unsigned*        object_id )
{  
  uint4 surface_info0      = surfacedatabuffer[surface_data_id    ];
  uint4 surface_info1      = surfacedatabuffer[surface_data_id + 1];
  uint4 surface_info2      = surfacedatabuffer[surface_data_id + 2];  
                           
  *volume_data_id          = surface_info0.z;
  *attribute_data_id       = surface_info0.w;
                           
  *object_id               = surface_info1.x;
  *surface_mesh_id         = surface_info1.y;
  *adjacent_volume_data_id = surface_info1.z;
  *adjacent_attribute_id   = surface_info1.w;

  *surface_is_outer        = surface_info2.x == 1;
  *surface_order           = uint2_t ( surface_info2.y, surface_info2.z );
  *surface_type            = surface_info2.w;
                           
  float4 volume_info0      = volumedatabuffer [ (*volume_data_id)    ];  
  float4 volume_info1      = volumedatabuffer [ (*volume_data_id) + 1];  
                           
  *volume_points_id        = floatBitsToInt(volume_info0.x);
  *attribute_points_id     = floatBitsToInt(attributedatabuffer[*attribute_data_id].z);
                           
  *adjacent_bbox_size      = volumedatabuffer [ (*adjacent_volume_data_id) ].w;  
                           
  *volume_bbox_size        = volume_info0.w;
  *volume_order            = uint3_t ( floatBitsToInt(volume_info1.x),
                                       floatBitsToInt(volume_info1.y),
                                       floatBitsToInt(volume_info1.z) );
  
}

#endif

