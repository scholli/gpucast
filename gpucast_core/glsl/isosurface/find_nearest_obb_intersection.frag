/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : find_nearest_obb_intersection.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

///////////////////////////////////////////////////////////////////////////////
bool find_nearest_obb_intersection_simple ( in isamplerBuffer volumelistbuffer,
                                            in ivec4          node,
                                            in vec4           ray_entry, 
                                            in vec4           ray_direction, 
                                            in float          tmax,
                                            out vec4          intersect_nearest,
                                            out vec4          normal_nearest,
                                            out vec3          uvw_in_nearest,
                                            out vec3          uvw_out_nearest,
                                            out ivec4         intersection_info )
{
  int volume_list_id    = node.w;
  int limit_id          = node.y;
  int number_of_volumes = node.z;
  normal_nearest        = vec4(0.0);

  bool  intersects_an_obb       = false;
  float t_nearest               = INFINITY;
  intersection_info             = ivec4(0);

  // variables for intersection result
  vec4 obb_normal_in      = vec4(0.0);
  vec4 obb_normal_out     = vec4(0.0);

  vec4 obb_size           = vec4(0.0);

  vec4 intersection_in    = vec4(0.0);
  vec4 intersection_out   = vec4(0.0);

  float t_in              = 0.0;
  float t_out             = 0.0;

  vec3 uvw_in             = vec3(0.0);
  vec3 uvw_out            = vec3(0.0);

  for (int i = 0; i != number_of_volumes; ++i)
  {
    ivec4 volume_geometric_info = texelFetchBuffer(volumelistbuffer, volume_list_id + 2*i    );
    ivec4 volume_bound_info     = texelFetchBuffer(volumelistbuffer, volume_list_id + 2*i + 1);
    int obb_index               = volume_bound_info.y;

    bool intersects         = intersect_obb ( obb_index, boundingboxbuffer, ray_entry, ray_direction, intersection_in, intersection_out, uvw_in, uvw_out, t_in, t_out, obb_normal_in, obb_normal_out, obb_size);

    if (  intersects &&                                   /* intersects oriented bounding box */
          (t_in < t_nearest || !intersects_an_obb) )
    {
        t_nearest           = t_in;                                                  
        intersect_nearest   = intersection_in;
        intersects_an_obb   = true;
        normal_nearest      = obb_normal_in;
        uvw_in_nearest      = uvw_in;
        uvw_out_nearest     = uvw_out;
        intersection_info   = ivec4(volume_bound_info.x, volume_geometric_info.yzw); // id, order_u, order_v, order_W;
    }
  }

  return intersects_an_obb;
}



///////////////////////////////////////////////////////////////////////////////
bool find_nearest_obb_intersection_including_iso_value (  in isamplerBuffer volumelistbuffer,
                                                          in samplerBuffer  limitbuffer,
                                                          in ivec4          node,
                                                          in vec4           ray_entry, 
                                                          in vec4           ray_direction, 
                                                          in float          tmax,
                                                          in vec4           isovalue,
                                                          out float         t_nearest,
                                                          out vec3          uvw_nearest,
                                                          out vec4          intersect_nearest,
                                                          out float         t_exit,
                                                          out vec3          uvw_exit,
                                                          out vec4          intersect_exit,
                                                          out vec4          nearest_obb_size,
                                                          inout ivec4       last_volume_list1,
                                                          inout ivec4       last_volume_list2,
                                                          out ivec4         found_index )
{
  // initialize output variables
  uvw_nearest       = vec3(0.0);
  uvw_exit          = vec3(0.0);
  nearest_obb_size  = vec4(0.0);
  t_exit            = 0.0;
  found_index       = ivec4(0);

  int volume_list_id    = node.w;
  int limit_id          = node.y;
  int number_of_volumes = node.z;
  vec4 normal_nearest   = vec4(0.0);

  bool  intersects_an_obb       = false;
  t_nearest                     = INFINITY;

  for (int i = 0; i != number_of_volumes; ++i)
  {
    ivec4 volume_geometric_info = texelFetchBuffer(volumelistbuffer, volume_list_id + 2*i    );
    ivec4 volume_bound_info     = texelFetchBuffer(volumelistbuffer, volume_list_id + 2*i + 1);
    int obb_index               = volume_bound_info.y;
    int limit_index             = volume_bound_info.z;

    vec4 vol_min_iso_value = texelFetchBuffer(limitbuffer, limit_index    );
    vec4 vol_max_iso_value = texelFetchBuffer(limitbuffer, limit_index + 1);

    // determine if volume contains isovalue
    bool volume_contains_iso_value  = is_in_range ( isovalue, vol_min_iso_value, vol_max_iso_value);

    // intersect obb only if obb contains isovalue
    if ( volume_contains_iso_value )
    {
      // variables for intersection result
      vec4 obb_normal_in      = vec4(0.0);
      vec4 obb_normal_out     = vec4(0.0);

      vec4 obb_size           = vec4(0.0);

      vec4 intersection_in    = vec4(0.0);
      vec4 intersection_out   = vec4(0.0);

      vec3 uvw_in             = vec3(0.0);
      vec3 uvw_out            = vec3(0.0);

      float t_in              = 0.0;
      float t_out             = 0.0;

      bool intersects         = intersect_obb ( obb_index, boundingboxbuffer, ray_entry, ray_direction, intersection_in, intersection_out, uvw_in, uvw_out, t_in, t_out, obb_normal_in, obb_normal_out, obb_size);

      if (  intersects &&                                   /* intersects oriented bounding box */
           (t_in < t_nearest || !intersects_an_obb) &&      /* intersection is nearer or no intersection found yet */
            last_volume_list1.x != volume_geometric_info.x &&
            last_volume_list1.y != volume_geometric_info.x &&
            last_volume_list1.z != volume_geometric_info.x &&
            last_volume_list1.w != volume_geometric_info.x &&
            last_volume_list2.x != volume_geometric_info.x &&
            last_volume_list2.y != volume_geometric_info.x &&
            last_volume_list2.z != volume_geometric_info.x &&
            last_volume_list2.w != volume_geometric_info.x
           )
      {

//#define USE_TRILINEAR_INTERPOLATION

#ifdef  USE_TRILINEAR_INTERPOLATION
        uvw_nearest               = trilinear_interpolation ( boundingboxbuffer, obb_index + 11, uvw_in);  // tri-linear interpolation with real uvw values
        uvw_exit                  = trilinear_interpolation ( boundingboxbuffer, obb_index + 11, uvw_out);
#else
        uvw_nearest             = uvw_in;                                                         // no interpolation
        uvw_exit                = uvw_out;           
#endif
        
        intersect_nearest         = intersection_in;
        intersect_exit            = intersection_out;

        t_nearest                 = t_in;
        t_exit                    = t_out;

        nearest_obb_size          = obb_size;
        intersects_an_obb         = true;
        normal_nearest            = obb_normal_in;

        found_index               = volume_geometric_info;
      }
    }
  }

  if ( intersects_an_obb ) 
  {
    // re-queue list
    last_volume_list2.w         = last_volume_list2.z;
    last_volume_list2.z         = last_volume_list2.y;
    last_volume_list2.y         = last_volume_list2.x;
    last_volume_list2.x         = last_volume_list1.w;
    last_volume_list1.w         = last_volume_list1.z;
    last_volume_list1.z         = last_volume_list1.y;
    last_volume_list1.y         = last_volume_list1.x;
    last_volume_list1.x         = found_index.x;
  }

  return intersects_an_obb;
}


