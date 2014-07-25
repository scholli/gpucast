#ifndef LIB_GPUCAST_HORNER_VOLUME_H
#define LIB_GPUCAST_HORNER_VOLUME_H

#include "./local_memory_config.h"

///////////////////////////////////////////////////////////////////////////////
template <typename T, unsigned weight_component>
__device__ inline 
T horner_volume ( T const*      points,
                  unsigned      baseid,
                  uint3 const&  order,
                  float3 const& uvw )
{
  // helper like binomial coefficients and t^n
  float one_minus_u  = 1.0f - uvw.x;
  float one_minus_v  = 1.0f - uvw.y;
  float one_minus_w  = 1.0f - uvw.z;

  float bcw    = 1.0;
  float wn     = 1.0;
  T point;

  // evaluate using horner scheme
  for (int k = 0; k != order.z; ++k)
  {
    T pv;
    float bcv = 1.0;
    float vn  = 1.0;

    for (int j = 0; j != order.y; ++j)
    {
      T pu;
      float bcu = 1.0;
      float un  = 1.0;

      for (int i = 0; i != order.x; ++i)
      {
        if (i == 0) 
        { // first interpolation (1-u)^n    
            pu = points[baseid + j * order.x + k * order.x * order.y] * one_minus_u;
        } else {
          if (i == order.x - 1) { // last interpolation u^n
            pu = pu + un * uvw.x * points[baseid + i + j * order.x + k * order.x * order.y];
          } else {  // else follow regular horner scheme
            un  *= uvw.x;
            bcu *= (float)(order.x - i) / (float)(i);
            pu = (pu + un * bcu * points[baseid + i + j * order.x + k * order.x * order.y]) * one_minus_u;
          }
        }
      }

      if (j == 0) { // first interpolation (1-v)^n    
          pv = pu * one_minus_v;
      } else {
        if (j == order.y - 1) {
          pv = pv + vn * uvw.y * pu;
        } else {
          vn  *= uvw.y;
          bcv *= (float)(order.y - j) / (float)(j);
          pv = (pv + vn * bcv * pu) * one_minus_v;
        }
      }
    }

    if (k == 0) {  // first interpolation (1-w)^n
        point = pv * one_minus_w;
    } else {
      if (k == order.z - 1) {
        point = point + wn * uvw.z * pv;
      } else {
        wn  *= uvw.z;
        bcw *= (float)(order.z - k) / (float)(k);
        point= (point + wn * bcw * pv) * one_minus_w;
      }
    }
  }

  float* weight_ptr = (float*)&point;
  float weight = *(weight_ptr+weight_component);
  point = point / weight;

  return point;
}


///////////////////////////////////////////////////////////////////////////////
template <typename T, unsigned weight_component>
__device__ inline void 
horner_volume_derivatives ( T const*           points,
                            unsigned           baseid,
                            uint3 const&       order,
                            float3 const&      uvw,
                            T&                 p,
                            T&                 du,
                            T&                 dv,
                            T&                 dw )
{
  T point;

  // helper like binomial coefficients and t^n
  float one_minus_u  = 1.0f - uvw.x;
  float one_minus_v  = 1.0f - uvw.y;
  float one_minus_w  = 1.0f - uvw.z;

  float bcw0         = 1.0f;
  float wn0          = 1.0f;
  float bcw1         = 1.0f;
  float wn1          = 1.0f;
  
  T p000;
  T p001;
  T p010;
  T p011;
  T p100;
  T p101;
  T p110;
  T p111;

  // evaluate using horner scheme
  for (int k = 0; k != order.z; ++k)
  {
    //T pv;

    T pv00;
    T pv01;
    T pv10;
    T pv11;

    float bcv0 = 1.0f;
    float bcv1 = 1.0f;
    float vn0  = 1.0f;
    float vn1  = 1.0f;

    for (int j = 0; j != order.y; ++j)
    {
      //T pu;
      T pu0;
      T pu1;
      float bcu  = 1.0f;
      float un   = 1.0f;
      
      // first interpolation (1-u)^n    
      pu0 = points[baseid +     j * order.x + k * order.x * order.y] * one_minus_u;      
      pu1 = points[baseid + 1 + j * order.x + k * order.x * order.y] * one_minus_u;

      for (int i = 1; i < order.x-2; ++i)
      {
        // follow regular horner scheme
        un  *= uvw.x;
        bcu *= (((float)(order.x) - 1.0f - (float)(i)) / (float)(i));
        pu0 = (pu0 + un * bcu * points[baseid +     i + j * order.x + k * order.x * order.y]) * one_minus_u; 
        pu1 = (pu1 + un * bcu * points[baseid + 1 + i + j * order.x + k * order.x * order.y]) * one_minus_u; 
      }

      // last interpolation u^n
      pu0 = pu0 + un * uvw.x * points[baseid + order.x-2 + j * order.x + k * order.x * order.y];
      pu1 = pu1 + un * uvw.x * points[baseid + order.x-1 + j * order.x + k * order.x * order.y];

      //pu = (1.0 - uvw.x) * pu0 + uvw.x * pu1;
      //pu = mix(pu0, pu1, uvw.x);

      if (j == 0) { // first interpolation (1-v)^n    
        pv00 = pu0 * one_minus_v;
        pv10 = pu1 * one_minus_v;
      }

      if (j == 1) {
        pv01 = pu0 * one_minus_v;
        pv11 = pu1 * one_minus_v;
      }

      if (j == 1 && order.y > 3) {
        vn0  *= uvw.y;
        bcv0 *= (((float)(order.y) - 1.0f - (float)(j)) / (float)(j));
        pv00  = (pv00 + vn0 * bcv0 * pu0) * one_minus_v; 
        pv10  = (pv10 + vn0 * bcv0 * pu1) * one_minus_v; 
      }

      if (j == order.y - 2) {
        pv00 = pv00 + vn0 * uvw.y * pu0;
        pv10 = pv10 + vn0 * uvw.y * pu1;
      }

      if (j == order.y - 2 && order.y > 3) {
        vn1  *= uvw.y;
        bcv1 *= (((float)(order.y) - (float)(j)) / (float)(j-1));
        pv01  = (pv01 + vn1 * bcv1 * pu0) * one_minus_v; 
        pv11  = (pv11 + vn1 * bcv1 * pu1) * one_minus_v; 
      }

      if (j == order.y - 1) {
        pv01  = pv01 + vn1 * uvw.y * pu0;
        pv11  = pv11 + vn1 * uvw.y * pu1;
      }

      if ( j > 1 && j < order.y - 2 && order.y > 3) {
        vn0  *= uvw.y;
        vn1  *= uvw.y;
        bcv0 *= (((float)(order.y) - 1.0f - j) / (float)(j));
        bcv1 *= (((float)(order.y)        - j) / (float)(j-1));
        
        pv00  = (pv00 + vn0 * bcv0 * pu0) * one_minus_v; 
        pv10  = (pv10 + vn0 * bcv0 * pu1) * one_minus_v; 
        pv01  = (pv01 + vn1 * bcv1 * pu0) * one_minus_v; 
        pv11  = (pv11 + vn1 * bcv1 * pu1) * one_minus_v; 
      }
    }

    if (k == 0) { // first interpolation (1-w)^n    
      p000 = pv00 * one_minus_w;
      p100 = pv10 * one_minus_w;
      p010 = pv01 * one_minus_w;
      p110 = pv11 * one_minus_w;
    }

    if (k == 1) {
      p001 = pv00 * one_minus_w;
      p101 = pv10 * one_minus_w;
      p011 = pv01 * one_minus_w;
      p111 = pv11 * one_minus_w;
    }

    if (k == 1 && order.z > 3) {
      wn0  *= uvw.z;
      bcw0 *= (((float)(order.z) - 1.0f - k) / (float)(k));
      p000  = (p000 + wn0 * bcw0 * pv00) * one_minus_w; 
      p100  = (p100 + wn0 * bcw0 * pv10) * one_minus_w; 
      p010  = (p010 + wn0 * bcw0 * pv01) * one_minus_w; 
      p110  = (p110 + wn0 * bcw0 * pv11) * one_minus_w; 
    }

    if (k == order.z - 2) {
      p000 = p000 + wn0 * uvw.z * pv00;
      p100 = p100 + wn0 * uvw.z * pv10;
      p010 = p010 + wn0 * uvw.z * pv01;
      p110 = p110 + wn0 * uvw.z * pv11;
    }

    if (k == order.z - 2 && order.z > 3) {
      wn1  *= uvw.z;
      bcw1 *= (((float)(order.z) - k) / (float)(k-1));

      p001  = (p001 + wn1 * bcw1 * pv00) * one_minus_w; 
      p101  = (p101 + wn1 * bcw1 * pv10) * one_minus_w; 
      p011  = (p011 + wn1 * bcw1 * pv01) * one_minus_w; 
      p111  = (p111 + wn1 * bcw1 * pv11) * one_minus_w; 
    }

    if (k == order.z - 1) {
      p001 = p001 + wn1 * uvw.z * pv00;
      p101 = p101 + wn1 * uvw.z * pv10;
      p011 = p011 + wn1 * uvw.z * pv01;
      p111 = p111 + wn1 * uvw.z * pv11;
    }

    if ( k > 1 && k < order.z - 2 && order.z > 3) {
      wn0  *= uvw.z;
      wn1  *= uvw.z;
      bcw0 *= (((float)(order.z) - 1.0f - k) / (float)(k));
      bcw1 *= (((float)(order.z)        - k) / (float)(k-1));
      
      p000  = (p000 + wn0 * bcw0 * pv00) * one_minus_w; 
      p100  = (p100 + wn0 * bcw0 * pv10) * one_minus_w; 
      p010  = (p010 + wn0 * bcw0 * pv01) * one_minus_w; 
      p110  = (p110 + wn0 * bcw0 * pv11) * one_minus_w; 

      p001  = (p001 + wn1 * bcw1 * pv00) * one_minus_w; 
      p101  = (p101 + wn1 * bcw1 * pv10) * one_minus_w; 
      p011  = (p011 + wn1 * bcw1 * pv01) * one_minus_w; 
      p111  = (p111 + wn1 * bcw1 * pv11) * one_minus_w; 
    }
  }

  // evaluate for u leaving a linear patch dependending on v,w
  T vw00 = mix(p000, p100, uvw.x);
  T vw10 = mix(p010, p110, uvw.x);
  T vw01 = mix(p001, p101, uvw.x);
  T vw11 = mix(p011, p111, uvw.x);

  // evaluate for v leaving a linear patch dependending on u,w
  T uw00 = mix(p000, p010, uvw.y);
  T uw10 = mix(p100, p110, uvw.y);
  T uw01 = mix(p001, p011, uvw.y);
  T uw11 = mix(p101, p111, uvw.y);

  // evaluating v,w plane for v resulting in last linear interpolation in w -> to compute firs  t partial derivative in w
  T w0 = mix(vw00, vw10, uvw.y);
  T w1 = mix(vw01, vw11, uvw.y);

  // evaluating v,w plane for w resulting in last linear interpolation in v -> to compute first partial derivative in v
  T v0 = mix(vw00, vw01, uvw.z);
  T v1 = mix(vw10, vw11, uvw.z);

  // evaluating v,w plane for w resulting in last linear interpolation in v -> to compute first partial derivative in v
  T u0 = mix(uw00, uw01, uvw.z);
  T u1 = mix(uw10, uw11, uvw.z);

  // last interpolation and back projection to euclidian space
  point = mix(w0, w1, uvw.z);

  // M.S. Floater '91 :
  //
  //             w[0]{n-1}(t) * w[1]{n-1}(t)
  // P'(t) = n * --------------------------- * P[1]{n-1}(t) - P[0]{n-1}(t)
  //                     w[0]{n})^2
  //
  // 1. recalculate overwritten helping point P[0, n-1]
  // 2. project P[0, n-1] and P[1, n-1] into plane w=1
  // 3. use formula above to find out the correct length of P'(t)

  float* pw_ptr  = (float*)&point;
  float pw       = *(pw_ptr+weight_component);

  float* u0w_ptr = (float*)&u0;
  float u0w      = *(u0w_ptr+weight_component);
  float* v0w_ptr = (float*)&v0;
  float v0w      = *(v0w_ptr+weight_component);
  float* w0w_ptr = (float*)&w0;
  float w0w      = *(w0w_ptr+weight_component);

  float* u1w_ptr = (float*)&u1;
  float u1w      = *(u1w_ptr+weight_component);
  float* v1w_ptr = (float*)&v1;
  float v1w      = *(v1w_ptr+weight_component);
  float* w1w_ptr = (float*)&w1;
  float w1w      = *(w1w_ptr+weight_component);

  p = point/pw;
  du = (order.x - 1.0f) * ((u0w * u1w) / (pw * pw)) * (u1/u1w - u0/u0w);
  dv = (order.y - 1.0f) * ((v0w * v1w) / (pw * pw)) * (v1/v1w - v0/v0w);
  dw = (order.z - 1.0f) * ((w0w * w1w) / (pw * pw)) * (w1/w1w - w0/w0w);
}



#endif // LIB_GPUCAST_HORNER_VOLUME_H