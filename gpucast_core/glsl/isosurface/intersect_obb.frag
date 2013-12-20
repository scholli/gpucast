/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : intersect_obb.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

bool intersect_obb (in int            bbox_id,
                    in samplerBuffer  bbox_buffer,
                    in vec4           ray_origin,
                    in vec4           ray_direction,
                    out vec4          intersection_in,
                    out vec4          intersection_out,
                    out vec3          uvw_in,
                    out vec3          uvw_out,
                    out float         t_in,
                    out float         t_out,
                    out vec4          normal_in,
                    out vec4          normal_out,
                    out vec4          size )
{
  uvw_in  = vec3(0.0);
  uvw_out = vec3(0.0);

  // fetch transformation data
  vec4 m0     = texelFetchBuffer(boundingboxbuffer, bbox_id + 0);
  vec4 m1     = texelFetchBuffer(boundingboxbuffer, bbox_id + 1);
  vec4 m2     = texelFetchBuffer(boundingboxbuffer, bbox_id + 2);
  vec4 m3     = texelFetchBuffer(boundingboxbuffer, bbox_id + 3);

  vec4 im0    = texelFetchBuffer(boundingboxbuffer, bbox_id + 4);
  vec4 im1    = texelFetchBuffer(boundingboxbuffer, bbox_id + 5);
  vec4 im2    = texelFetchBuffer(boundingboxbuffer, bbox_id + 6);
  vec4 im3    = texelFetchBuffer(boundingboxbuffer, bbox_id + 7);

  // load orientation -> TODO -> Optimize -> translation and invert can be done at preprocessing
  mat4 O      = mat4(m0, m1, m2, m3);
  mat4 invO   = mat4(im0, im1, im2, im3);

  vec4 low    = texelFetchBuffer(boundingboxbuffer, bbox_id + 8);
  vec4 high   = texelFetchBuffer(boundingboxbuffer, bbox_id + 9);
  vec4 center = texelFetchBuffer(boundingboxbuffer, bbox_id + 10);

  size        = high - low;

  // transform origin and direction to local bounding box coordinates
  vec4 ray_origin_local    = invO * vec4(ray_origin.xyz - center.xyz, 1.0);
  vec4 ray_direction_local = invO * vec4(ray_direction.xyz, 0.0);

  // compute all potential intersections with all bounding planes
  vec3 tlow  = (low.xyz  - ray_origin_local.xyz) / ray_direction_local.xyz;
  vec3 thigh = (high.xyz - ray_origin_local.xyz) / ray_direction_local.xyz;
  vec3 tmin  = min(tlow, thigh);
  vec3 tmax  = max(tlow, thigh);

  vec4 potential_intersect  = vec4(0.0);
  bool found_in             = false;
  bool found_out            = false;
  t_in                      = INFINITY;
  t_out                     = INFINITY;

  // check intersections of bounding box
  potential_intersect = ray_origin_local + tmin.x * ray_direction_local;
  if ( potential_intersect[1] > low[1] && potential_intersect[1] < high[1] &&
       potential_intersect[2] > low[2] && potential_intersect[2] < high[2] ) 
  {
    intersection_in     = ray_origin + tmin.x * ray_direction;                   // compute intersection point in object space
    uvw_in              = (potential_intersect.xyz - low.xyz) / (high.xyz - low.xyz);  // map intersection to uvw[0,1] domain -> heuristic for uvw-parameter
    t_in                = tmin.x;
    normal_in           = O * vec4(1.0, 0.0, 0.0, 0.0);
    found_in  = true;
  }

  potential_intersect = ray_origin_local + tmin.y * ray_direction_local;
  if ( potential_intersect[0] > low[0] && potential_intersect[0] < high[0] &&
       potential_intersect[2] > low[2] && potential_intersect[2] < high[2] ) 
  {
    intersection_in     = ray_origin + tmin.y * ray_direction;                   // compute intersection point in object space
    uvw_in              = (potential_intersect.xyz - low.xyz) / (high.xyz - low.xyz);  // map intersection to uvw[0,1] domain -> heuristic for uvw-parameter
    t_in                = tmin.y;
    normal_in           = O * vec4(0.0, -1.0, 0.0, 0.0);
    found_in  = true;
  }

  potential_intersect = ray_origin_local + tmin.z * ray_direction_local;
  if ( potential_intersect[0] > low[0] && potential_intersect[0] < high[0] &&
       potential_intersect[1] > low[1] && potential_intersect[1] < high[1] )
  {
    intersection_in     = ray_origin + tmin.z * ray_direction;                   // compute intersection point in object space
    uvw_in              = (potential_intersect.xyz - low.xyz) / (high.xyz - low.xyz);  // map intersection to uvw[0,1] domain -> heuristic for uvw-parameter
    t_in                = tmin.z;
    normal_in           = O * vec4(0.0, 0.0, -1.0, 0.0);
    found_in  = true;
  }

  potential_intersect = ray_origin_local + tmax.x * ray_direction_local;
  if ( potential_intersect[1] > low[1] && potential_intersect[1] < high[1] &&
       potential_intersect[2] > low[2] && potential_intersect[2] < high[2]) 
  {
    intersection_out    = ray_origin + tmax.x * ray_direction;                   // compute intersection point in object space
    uvw_out             = (potential_intersect.xyz - low.xyz) / (high.xyz - low.xyz);  // map intersection to uvw[0,1] domain -> heuristic for uvw-parameter
    t_out               = tmax.x;
    normal_out           = O * vec4(-1.0, 0.0, 0.0, 0.0);
    found_out  = true;
  }
  
  potential_intersect = ray_origin_local + tmax.y * ray_direction_local;
  if ( potential_intersect[0] > low[0] && potential_intersect[0] < high[0] &&
       potential_intersect[2] > low[2] && potential_intersect[2] < high[2]) 
  {
    intersection_out    = ray_origin + tmax.y * ray_direction;                   // compute intersection point in object space
    uvw_out             = (potential_intersect.xyz - low.xyz) / (high.xyz - low.xyz);  // map intersection to uvw[0,1] domain -> heuristic for uvw-parameter
    t_out               = tmax.y;
    normal_out           = O * vec4(0.0, 1.0, 0.0, 0.0);
    found_out  = true;
  }
 
  potential_intersect = ray_origin_local + tmax.z * ray_direction_local;
  if ( potential_intersect[0] > low[0] && potential_intersect[0] < high[0] &&
       potential_intersect[1] > low[1] && potential_intersect[1] < high[1]) 
  {
    intersection_out    = ray_origin + tmax.z * ray_direction;                   // compute intersection point in object space
    uvw_out             = (potential_intersect.xyz - low.xyz) / (high.xyz - low.xyz);  // map intersection to uvw[0,1] domain -> heuristic for uvw-parameter
    t_out               = tmax.z;
    normal_out           = O * vec4(0.0, 0.0, 1.0, 0.0);
    found_out  = true;
  }

  return found_in && found_out;
}

