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

layout(std430) buffer hullvertexmap {
  hull_vertex_entry hvm[];
};

float calculate_obb_area(mat4           modelview_projection,
                         mat4           modelview_inverse,
                         vec4           obb_center,
                         mat4           obb_orientation,
                         mat4           obb_orientation_inverse,
                         samplerBuffer  obb_vertices,
                         int            obb_base_index )
{
  // transform eye to obb space
  vec4 eye_object_space = modelview_inverse * vec4(0.0, 0.0, 0.0, 1.0);
  vec4 eye_obb_space = obb_orientation_inverse * vec4(eye_object_space.xyz - obb_center.xyz, 1.0);

  // copy obb vertices to local array
  vec4 bbox[8];
  for(int i = 0; i != 8; ++i) {
    bbox[i] = texelFetch(obb_vertices, obb_base_index + i);
  }

  // identify in which quadrant the eye is located
  float sum = 0.0;

  int pos = (int(eye_obb_space.x < bbox[0].x))        //  1 = left   |  compute 6-bit
          + (int(eye_obb_space.x > bbox[7].x) << 1)   //  2 = right  |        code to
          + (int(eye_obb_space.y < bbox[0].y) << 2)   //  4 = bottom |   classify eye
          + (int(eye_obb_space.y > bbox[7].y) << 3)   //  8 = top    |with respect to
          + (int(eye_obb_space.z < bbox[0].z) << 4)   // 16 = front  | the 6 defining
          + (int(eye_obb_space.z > bbox[7].z) << 5);  // 32 = back   |         planes

  // look up according number of visible vertices
  int n_visible_vertices = int(hvm[pos].num_visible_vertices);
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
  return abs(sum); 
}

#endif