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
  hull_vertex_entry data[];
};

#if 0

float calculate_obb_area(vec3           eye,
                         mat4           mvp,
                         samplerBuffer  obb_vertices,
                         int            obb_base_index )
{
  vec3 bbox[8];
  for(int i = 0; i != 8; ++i)

  Vector2D dst[8]; 
  float sum = 0.0; 
  int pos;
  
  
  vec3 bbox_min = texelFetch(obb_vertices, obb_base_index).xyz;
  vec3 bbox_max = texelFetch(obb_vertices, obb_base_index + 7).xyz;

  int pos = ((eye.x < bbox_min.x)     )   //  1 = left   |  compute 6-bit
          + ((eye.x > bbox_max.x) << 1)   //  2 = right  |        code to
          + ((eye.y < bbox_min.y) << 2)   //  4 = bottom |   classify eye
          + ((eye.y > bbox_max.y) << 3)   //  8 = top    |with respect to
          + ((eye.z < bbox_min.z) << 4)   // 16 = front  | the 6 defining
          + ((eye.z > bbox_max.z) << 5);  // 32 = back   |         planes

  int num = texelFetch()

  if (!num = hullvertex[pos][6]) 
    return 0.0;  //look up number of vertices

  for(int i=0; i<num; i++) 
    dst[i] := projectToScreen(bbox[hullvertex[pos][i]]);

  for(int i=0; i<num; i++) 
    sum += (dst[ i ].x - dst[ (i+1) % num ].x) * (dst[ i ].y + dst[ (i+1) % num ].y);

  return sum * 0.5;                               //return corrected value
}

#endif

#endif