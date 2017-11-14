/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_multipass/hull_pass.geom
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_EXT_geometry_shader4: enable

/********************************************************************************
* constants
********************************************************************************/
precision highp float;

/********************************************************************************
* uniforms
********************************************************************************/
uniform mat4          modelviewprojectionmatrix;
uniform mat4          modelviewmatrix;
uniform mat4          modelviewmatrixinverse;
uniform mat4          normalmatrix;

uniform bool          discard_by_minmax;
uniform float         threshold;

uniform usamplerBuffer surfacedatabuffer;
uniform samplerBuffer  volumedatabuffer;
uniform samplerBuffer  attributedatabuffer;

/********************************************************************************
* input
********************************************************************************/
layout(triangles) in;

in vec4      vertex_position[3];
in vec3      vertex_parameter[3];
flat in uint vertex_surface_id[3];

/********************************************************************************
* output
********************************************************************************/
layout(max_vertices = 3) out;

out vec4        fragposition;
out vec3        parameter;
flat out uvec4  surface_info;  // surface_id,  volume_id,  attribute_id, is_outer
flat out uvec4  volume_info;   // surface_uid, volume_uid, contains_isosurface, adjacent_contains_isosurface

/********************************************************************************
* functions
********************************************************************************/
#include "resources/glsl/isosurface/target_function.frag"

/********************************************************************************
* pass only triangles that match minmax range of iso value
********************************************************************************/
void main(void)
{
  // retrieve data from texture memory
  uvec4 surface_info0_as_float   = texelFetchBuffer ( surfacedatabuffer, int(vertex_surface_id[0]    ) );
  uvec4 surface_info1_as_float   = texelFetchBuffer ( surfacedatabuffer, int(vertex_surface_id[0] + 1) );
  uvec4 surface_info2_as_float   = texelFetchBuffer ( surfacedatabuffer, int(vertex_surface_id[0] + 2) );

  // retrieve ids and if the surface belongs to the outer hull 
  uint unique_id                 = surface_info0_as_float.x;
  uint attribute_data_id         = surface_info0_as_float.w;
  uint adj_attribute_data_id     = surface_info1_as_float.w;
  bool is_outer_face             = surface_info2_as_float.x != 0;
  bool is_outer_cell             = surface_info1_as_float.x != 0;

  // determine if volume needs to be rendered
  float attribute_min            = texelFetchBuffer ( attributedatabuffer, int(attribute_data_id) ).x;
  float attribute_max            = texelFetchBuffer ( attributedatabuffer, int(attribute_data_id) ).y;
  bool  volume_contains_isovalue = is_in_range ( threshold, attribute_min, attribute_max );

  float adj_attribute_min        = texelFetchBuffer ( attributedatabuffer, int(adj_attribute_data_id) ).x;
  float adj_attribute_max        = texelFetchBuffer ( attributedatabuffer, int(adj_attribute_data_id) ).y;
  bool  adj_volume_contains_isovalue  = is_in_range ( threshold, adj_attribute_min, adj_attribute_max );

  // only pass surfaces that are adjacent to volumes potentially containing an isovalue
  if ( is_outer_face || is_outer_cell || adj_volume_contains_isovalue || volume_contains_isovalue )
  {
    for ( int i = 0; i < 3; ++i )
    {
        gl_Position     = gl_PositionIn[i];
    
        fragposition    = vertex_position[i];
        parameter       = vertex_parameter[i];
        surface_info    = uvec4(vertex_surface_id[i], vertex_surface_id[i], vertex_surface_id[i], vertex_surface_id[i]);
        volume_info     = uvec4(vertex_surface_id[i], vertex_surface_id[i], vertex_surface_id[i], vertex_surface_id[i]);

        EmitVertex();
    }
    EndPrimitive();
  }
}

