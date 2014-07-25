/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : sample_volume.h
*  project    : gpucast
*  description:
*
********************************************************************************/

#define float2_t (float2)
#define float3_t (float3)
#define float4_t (float4)

#define uint2_t  (uint2)
#define uint3_t  (uint3)
#define uint4_t  (uint4)


///////////////////////////////////////////////////////////////////////////////
float4 euclidian_space(float4 p_homogenous)
{
  p_homogenous.x /= p_homogenous.w;
  p_homogenous.y /= p_homogenous.w;
  p_homogenous.w /= p_homogenous.w;

  return p_homogenous;
}


///////////////////////////////////////////////////////////////////////////////
float4 evaluate_volume ( int              order_u,
                         int              order_v,
                         int              order_w,
                         float            u,
                         float            v,
                         float            w,
                         __local  float4* points
                       )
{
  // helper like binomial coefficients and t^n
  float one_minus_u  = 1 - u;
  float one_minus_v  = 1 - v;
  float one_minus_w  = 1 - w;

  float bcw    = 1;
  float wn     = 1;
  float4 point = 0;

  // evaluate using horner scheme
  for (int k = 0; k != order_w; ++k)
  {
    float4 pv = 0;
    float bcv = 1;
    float vn  = 1;

    for (int j = 0; j != order_v; ++j)
    {
      float4 pu = 0;
      float bcu = 1;
      float un  = 1;

      for (int i = 0; i != order_u; ++i)
      {
        if (i == 0) 
        { // first interpolation (1-u)^n    
            pu = points[j * order_u + k * order_u * order_v] * one_minus_u;
        } else {
          if (i == order_u - 1) { // last interpolation u^n
            pu += un * u * points[i + j * order_u + k * order_u * order_v];
          } else {  // else follow regular horner scheme
            un  *= u;
            bcu *= (float)(order_u - i) / (float)(i);
            pu = (pu + un * bcu * points[i + j * order_u + k * order_u * order_v]) * one_minus_u;
          }
        }
      }

      if (j == 0) { // first interpolation (1-v)^n    
          pv = pu * one_minus_v;
      } else {
        if (j == order_v - 1) {
          pv += vn * v * pu;
        } else {
          vn  *= v;
          bcv *= (float)(order_v - j) / (float)(j);
          pv = (pv + vn * bcv * pu) * one_minus_v;
        }
      }
    }

    if (k == 0) {  // first interpolation (1-w)^n
        point = pv * one_minus_w;
    } else {
      if (k == order_w-1) {
        point += wn * w * pv;
      } else {
        wn  *= w;
        bcw *= (float)(order_w - k) / (float)(k);
        point= (point + wn * bcw * pv) * one_minus_w;
      }
    }
  }

  return point;
}



///////////////////////////////////////////////////////////////////////////////
__kernel void run_kernel ( int              order_u, 
                           int              order_v,
                           int              order_w,
                           __global float4* points, 
                           __global float4* samples,
                           int              points_offset,
                           int              samples_offset,
                           float            scale )
{
  int gid_x = get_group_id (0);
  int gid_y = get_group_id (1);
  int gid_z = get_group_id (2);

  int lid_x = get_local_id (0);
  int lid_y = get_local_id (1);
  int lid_z = get_local_id (2);

  int size_x = get_global_size (0);
  int size_y = get_global_size (1);
  int size_z = get_global_size (2);

  int groupsize_x = get_local_size (0);
  int groupsize_y = get_local_size (1);
  int groupsize_z = get_local_size (2);

  float u = (float)(gid_x * groupsize_x + lid_x) / (size_x - 1);
  float v = (float)(gid_y * groupsize_y + lid_y) / (size_y - 1);
  float w = (float)(gid_z * groupsize_z + lid_z) / (size_z - 1);

  int local_index = lid_z * groupsize_x * groupsize_y +
                    lid_y * groupsize_x +
                    lid_x;

  // create local storage of control points
  __local float4 local_points[4*4*4];
  local_points[local_index] = points[points_offset + local_index];

  int global_index =  size_x * size_y * (gid_z * groupsize_z + lid_z) +
                      size_x *          (gid_y * groupsize_y + lid_y) +
                                        (gid_x * groupsize_x + lid_x);
  
  samples[samples_offset + global_index] = scale * evaluate_volume(order_u, order_v, order_w, u, v, w, local_points);

}