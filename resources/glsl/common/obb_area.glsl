#ifndef GPUCAST_OBB_AREA
#define GPUCAST_OBB_AREA

/********************************************************************************
*
* Copyright (C) 2016 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : obb_area.glsl
*  project    : gpucast
*  description:
*
********************************************************************************/

struct hull_vertex_entry {
  unsigned char id;
  unsigned char num_visible_vertices;
  unsigned char vertices[6];
};

layout(std430, binding = GPUCAST_HULLVERTEXMAP_SSBO_BINDING) buffer gpucast_hullvertexmap_ssbo {
  hull_vertex_entry gpucast_hvm[];
};

///////////////////////////////////////////////////////////////////////////////
void fetch_obb_data(in samplerBuffer obb_data, in int base_id, out vec4 obb_center, out mat4 obb_orientation, out mat4 obb_orientation_inverse, out vec4 obb_vertices[8])
{
  obb_center = texelFetch(obb_data, base_id);

  obb_orientation = mat4(texelFetch(obb_data, base_id + 3),
                              texelFetch(obb_data, base_id + 4),
                              texelFetch(obb_data, base_id + 5),
                              texelFetch(obb_data, base_id + 6));

  obb_orientation_inverse = mat4(texelFetch(obb_data, base_id + 7),
                                 texelFetch(obb_data, base_id + 8),
                                 texelFetch(obb_data, base_id + 9),
                                 texelFetch(obb_data, base_id + 10));
  for (int i = 0; i != 8; ++i) {
    obb_vertices[i] = texelFetch(obb_data, base_id + 11 + i);
  }
}

///////////////////////////////////////////////////////////////////////////////
float calculate_obb_area(mat4           modelview_projection,
                         mat4           modelview_inverse,
                         samplerBuffer  obb_data,
                         int            obb_base_index )
{
  vec4 obb_center;
  mat4 obb_orientation;
  mat4 obb_orientation_inverse;
  vec4 bbox[8];

  fetch_obb_data(obb_data, obb_base_index, obb_center, obb_orientation, obb_orientation_inverse, bbox);

  // transform eye to obb space
  vec4 eye_object_space = modelview_inverse * vec4(0.0, 0.0, 0.0, 1.0);
  vec4 eye_obb_space = obb_orientation_inverse * vec4(eye_object_space.xyz - obb_center.xyz, 1.0);

  // identify in which quadrant the eye is located
  float sum = 0.0;

  int pos = (int(eye_obb_space.x < bbox[0].x))        //  1 = left   |  compute 6-bit
          + (int(eye_obb_space.x > bbox[6].x) << 1)   //  2 = right  |        code to
          + (int(eye_obb_space.y < bbox[0].y) << 2)   //  4 = bottom |   classify eye
          + (int(eye_obb_space.y > bbox[6].y) << 3)   //  8 = top    |with respect to
          + (int(eye_obb_space.z < bbox[0].z) << 4)   // 16 = front  | the 6 defining
          + (int(eye_obb_space.z > bbox[6].z) << 5);  // 32 = back   |         planes

  // look up according number of visible vertices
  int n_visible_vertices = int(gpucast_hvm[pos].num_visible_vertices);
  if (n_visible_vertices == 0) {
    return 0.0;
  }

  // project all obb vertices to screen coordinates
  vec2 dst[6];
  for (int i = 0; i != n_visible_vertices; ++i) {
    vec4 corner_screenspace = modelview_projection * (obb_orientation * bbox[i] + vec4(obb_center.xyz, 0.0));
    corner_screenspace /= corner_screenspace.w;
    dst[i] = clamp(corner_screenspace.xy, vec2(-1.0), vec2(1.0));
  }

  // accumulate area of visible vertices' polygon
  for (int i = 0; i < n_visible_vertices; i++) {
    sum += (dst[i].x - dst[(i + 1) % n_visible_vertices].x) * (dst[i].y + dst[(i + 1) % n_visible_vertices].y);
  }

  // return area
  return abs(sum) / 4.0; // this differs from original, but testet with extra application
}

#endif