#ifndef LIB_GPUCAST_HORNER_SURFACE_H
#define LIB_GPUCAST_HORNER_SURFACE_H

#include <math/mix.h>

/*******************************************************************************
 * Evaluate Surface using modificated horner algorithm in Bernstein basis
 * points assumed in homogenous coordinates! :  p = [wx wy wz w]
*******************************************************************************/
__device__
inline float4
horner_surface ( float4 const*     points,
                 unsigned          baseid,
                 uint2 const&      order,
                 float2 const&     uv ) 
{
  float4 point = float4_t(0.0f, 0.0f, 0.0f, 0.0f);

  float bcu     = 1.0f;
  float un      = 1.0f;
  int deg_u     = order.x - 1;
  int deg_v     = order.y - 1;

  float4 u0_0 = points[baseid              ] * (1.0f - uv.x);
  float4 u0_1 = points[baseid + 1          ] * (1.0f - uv.x);
  float4 u1_0 = points[baseid + order.x    ] * (1.0f - uv.x);
  float4 u1_1 = points[baseid + order.x + 1] * (1.0f - uv.x);

  /**************************************** 1. step : horner for first 2 rows *********************************/
  int i = 0;
  for (i = 1; i <= deg_u - 2; ++i) 
  {
    un  = un * uv.x;
    bcu = (bcu * (float)(deg_u - i)) / (float)(i);

    u0_0 = (u0_0 + un * bcu * points[baseid + i    ])           * (1.0f - uv.x);
    u0_1 = (u0_1 + un * bcu * points[baseid + i + 1])           * (1.0f - uv.x);

    u1_0 = (u1_0 + un * bcu * points[baseid + order.x + i    ]) * (1.0f - uv.x);
    u1_1 = (u1_1 + un * bcu * points[baseid + order.x + i + 1]) * (1.0f - uv.x);
  }

  u0_0 = u0_0 + un * uv.x * points[baseid +           deg_u - 1];
  u0_1 = u0_1 + un * uv.x * points[baseid +           deg_u    ];
  u1_0 = u1_0 + un * uv.x * points[baseid + order.x + deg_u - 1];
  u1_1 = u1_1 + un * uv.x * points[baseid + order.x + deg_u    ]; 
  
  /* point in first and second row */
  float4 u0 = (1.0f - uv.x) * u0_0 + uv.x * u0_1;
  float4 u1 = (1.0f - uv.x) * u1_0 + uv.x * u1_1; 

  /**************************************** 2. step : inner loop for rows 3 to order - 1 ***********************/
  float bcv = 1.0f;
  float vn = 1.0f;

  float4 v0 = u0 * (1.0f - uv.y);
  float4 v1 = u1 * (1.0f - uv.y);

  float4 ui, ui_0, ui_1;
  for (i = 1; i <= deg_v - 2; ++i) 
  {
    bcu = 1.0f;
    un = 1.0f;
    ui_0 = points[baseid + (i+1) * order.x    ] * (1.0f - uv.x);
    ui_1 = points[baseid + (i+1) * order.x + 1] * (1.0f - uv.x);

    int j;
    for (j = 1; j <= deg_u-2; ++j) {
      un = un * uv.x;
      bcu = bcu * (float)(deg_u - j) / (float)(j);
      ui_0 = (ui_0 + un * bcu * points[baseid + (i+1) * order.x + j    ]) * (1.0f - uv.x);
      ui_1 = (ui_1 + un * bcu * points[baseid + (i+1) * order.x + j + 1]) * (1.0f - uv.x);
    }
    ui_0 = ui_0 + un * uv.x * points[baseid + (i+1) * order.x + deg_u - 1];
    ui_1 = ui_1 + un * uv.x * points[baseid + (i+1) * order.x + deg_u    ];
    ui = (1.0f - uv.x) * ui_0 + uv.x * ui_1;

    u0 = u1;
    u1 = ui;
    
    vn = vn * uv.y;
    bcv = bcv * (float)(deg_v - i) / (float)(i);
    v0 = (v0 + vn * bcv * u0) * (1.0f - uv.y);
    v1 = (v1 + vn * bcv * u1) * (1.0f - uv.y);
  }
  
  /**************************************** 3. step : horner scheme for last row *******************************/
  bcu = 1.0f; 
  un = 1.0f; 
  ui_0 = points[baseid + deg_v * order.x    ] * (1.0f - uv.x); 
  ui_1 = points[baseid + deg_v * order.x + 1] * (1.0f - uv.x);

  for (i = 1; i <= deg_u-2; ++i) 
  { 
    un = un * uv.x; 
    bcu = bcu * (float)(deg_u-i) / (float)(i); 
    ui_0 = (ui_0 + un * bcu * points[baseid + deg_v * order.x + i    ]) * (1.0f - uv.x);
    ui_1 = (ui_1 + un * bcu * points[baseid + deg_v * order.x + i + 1]) * (1.0f - uv.x);
  }

  ui_0 = ui_0 + un * uv.x * points[baseid + deg_v * order.x + deg_u - 1];
  ui_1 = ui_1 + un * uv.x * points[baseid + deg_v * order.x + deg_u    ];
  //ui = (1.0f - uv.x) * ui_0 + uv.x * ui_1;
  ui = mix(ui_0, ui_1, uv.x);
  
  
  /**************************************** 4. step : final interpolation over v ********************************/
  v0 = v0 + vn * uv.y * u1;
  v1 = v1 + vn * uv.y * ui;

  //point = (1.0f - uv.y) * v0 + uv.y * v1;
  point = mix (v0, v1, uv.y);
  point = point/point.w;  

  return point;
}



/*******************************************************************************
 * Evaluate Surface using modificated horner algorithm in Bernstein basis
 * points assumed in homogenous coordinates! :  p = [wx wy wz w]
*******************************************************************************/
__device__
inline void 
horner_surface_derivatives ( float4 const*     points,
                             unsigned          baseid,
                             uint2 const&      order,
                             float2 const&     uv,
                             float4&           p, 
                             float4&           du, 
                             float4&           dv) 
{
  float4 point;

  float bcu = 1.0f;
  float un  = 1.0f;

  int deg_u = order.x - 1;
  int deg_v = order.y - 1;

  float4 u0_0 = points[baseid              ] * (1.0f - uv.x);
  float4 u0_1 = points[baseid + 1          ] * (1.0f - uv.x);
  float4 u1_0 = points[baseid + order.x    ] * (1.0f - uv.x);
  float4 u1_1 = points[baseid + order.x + 1] * (1.0f - uv.x);

  /**************************************** 1. step : horner for first 2 rows *********************************/
  int i;
  for (i = 1; i <= deg_u - 2; ++i) {
    un = un * uv.x;
    bcu = bcu * (float)(deg_u - i) / (float)(i);

    u0_0 = (u0_0 + un * bcu * points[baseid + i    ])           * (1.0f - uv.x);
    u0_1 = (u0_1 + un * bcu * points[baseid + i + 1])           * (1.0f - uv.x);

    u1_0 = (u1_0 + un * bcu * points[baseid + order.x + i    ]) * (1.0f - uv.x);
    u1_1 = (u1_1 + un * bcu * points[baseid + order.x + i + 1]) * (1.0f - uv.x);
  }

  u0_0 = u0_0 + un * uv.x * points[baseid +           deg_u - 1];
  u0_1 = u0_1 + un * uv.x * points[baseid +           deg_u    ];
  u1_0 = u1_0 + un * uv.x * points[baseid + order.x + deg_u - 1];
  u1_1 = u1_1 + un * uv.x * points[baseid + order.x + deg_u    ]; 
  
  /* point in first and second row */
  float4 u0 = (1.0f - uv.x) * u0_0 + uv.x * u0_1;
  float4 u1 = (1.0f - uv.x) * u1_0 + uv.x * u1_1; 

  /**************************************** 2. step : inner loop for rows 3 to order - 1 ***********************/
  float bcv = 1.0f;
  float vn = 1.0f;

  float4 v0 = u0 * (1.0f - uv.y);
  float4 v1 = u1 * (1.0f - uv.y);

  float4 ui, ui_0, ui_1;
  for (i = 1; i <= deg_v - 2; ++i) 
  {
    bcu = 1.0f;
    un = 1.0f;
    ui_0 = points[baseid + (i+1) * order.x    ] * (1.0f - uv.x);
    ui_1 = points[baseid + (i+1) * order.x + 1] * (1.0f - uv.x);

    int j;
    for (j = 1; j <= deg_u-2; ++j) 
    {
      un = un * uv.x;
      bcu = bcu * (float)(deg_u - j) / (float)(j);
      ui_0 = (ui_0 + un * bcu * points[baseid + (i+1) * order.x + j    ]) * (1.0f - uv.x);
      ui_1 = (ui_1 + un * bcu * points[baseid + (i+1) * order.x + j + 1]) * (1.0f - uv.x);
    }
    ui_0 = ui_0 + un * uv.x * points[baseid + (i+1) * order.x + deg_u - 1];
    ui_1 = ui_1 + un * uv.x * points[baseid + (i+1) * order.x + deg_u    ];
    ui = (1.0f - uv.x) * ui_0 + uv.x * ui_1;

    u0 = u1;
    u1 = ui;
    
    vn = vn * uv.y;
    bcv = bcv * (float)(deg_v - i) / (float)(i);
    v0 = (v0 + vn * bcv * u0) * (1.0f - uv.y);
    v1 = (v1 + vn * bcv * u1) * (1.0f - uv.y);
  }
  
  /**************************************** 3. step : horner scheme for last row *******************************/
  bcu = 1.0f; 
  un = 1.0f; 
  ui_0 = points[baseid + deg_v * order.x    ] * (1.0f - uv.x); 
  ui_1 = points[baseid + deg_v * order.x + 1] * (1.0f - uv.x);

  for (i = 1; i <= deg_u-2; ++i) 
  { 
    un = un * uv.x; 
    bcu = bcu * (float)(deg_u-i) / (float)(i); 
    ui_0 = (ui_0 + un * bcu * points[baseid + deg_v * order.x + i    ]) * (1.0f - uv.x);
    ui_1 = (ui_1 + un * bcu * points[baseid + deg_v * order.x + i + 1]) * (1.0f - uv.x);
  }

  ui_0 = ui_0 + un * uv.x * points[baseid + deg_v * order.x + deg_u - 1];
  ui_1 = ui_1 + un * uv.x * points[baseid + deg_v * order.x + deg_u    ];
  ui = (1.0f - uv.x) * ui_0 + uv.x * ui_1;
  
  /**************************************** 4. step : final interpolation over v ********************************/
  v0 = v0 + vn * uv.y * u1;
  v1 = v1 + vn * uv.y * ui;

  point = (1.0f - uv.y) * v0 + uv.y * v1;

  /* transform to euclidian space */
  dv = (order.y - 1) * ((v0.w * v1.w) / (point.w * point.w)) * ((v1 / v1.w) - (v0 / v0.w));

  /************************************ 5.step : dartial derivative over u ***********************************/
  bcv = 1.0f;
  vn = 1.0f;

  float4 v0_0 = points[baseid              ] * (1.0f - uv.y);
  float4 v0_1 = points[baseid + order.x    ] * (1.0f - uv.y);
  float4 v1_0 = points[baseid + 1          ] * (1.0f - uv.y);
  float4 v1_1 = points[baseid + order.x + 1] * (1.0f - uv.y);

  for (i = 1; i <= deg_v - 2; ++i) 
  {
    vn = vn * uv.y;
    bcv = bcv * (float)(deg_v - i) / (float)(i);

    v0_0 = (v0_0 + vn * bcv * points[baseid + (i  ) * order.x    ]) * (1.0f - uv.y);
    v0_1 = (v0_1 + vn * bcv * points[baseid + (i+1) * order.x    ]) * (1.0f - uv.y);

    v1_0 = (v1_0 + vn * bcv * points[baseid + (i  ) * order.x + 1]) * (1.0f - uv.y);
    v1_1 = (v1_1 + vn * bcv * points[baseid + (i+1) * order.x + 1]) * (1.0f - uv.y);
  }

  v0_0 = v0_0 + vn * uv.y * points[baseid + (deg_v-1) * order.x    ];
  v0_1 = v0_1 + vn * uv.y * points[baseid + (deg_v  ) * order.x    ];
  v1_0 = v1_0 + vn * uv.y * points[baseid + (deg_v-1) * order.x + 1];
  v1_1 = v1_1 + vn * uv.y * points[baseid + (deg_v  ) * order.x + 1]; 
  
  /* point in first and second row */
  v0 = (1.0f - uv.y) * v0_0 + uv.y * v0_1;
  v1 = (1.0f - uv.y) * v1_0 + uv.y * v1_1; 

  /*********************************** 6. step : for all columns *******************************************/
  bcu = 1.0f;
  un = 1.0f;

  u0 = v0 * (1.0f - uv.x);
  u1 = v1 * (1.0f - uv.x);

  float4 vi_0, vi_1, vi;
  for (i = 1; i <= deg_u - 2; ++i) 
  {
    bcv = 1.0f;
    vn = 1.0f;
    vi_0 = points[baseid +           i + 1] * (1.0f - uv.y);
    vi_1 = points[baseid + order.x + i + 1] * (1.0f - uv.y);

    int j;
    for (j = 1; j <= deg_v-2; ++j) 
    {
      vn = vn * uv.y;
      bcv = bcv * (float)(deg_v - j) / (float)(j);
      vi_0 = (vi_0 + vn * bcv * points[baseid + (j  ) * order.x + i + 1]) * (1.0f - uv.y);
      vi_1 = (vi_1 + vn * bcv * points[baseid + (j+1) * order.x + i + 1]) * (1.0f - uv.y);
    }
    vi_0 = vi_0 + vn * uv.y * points[baseid + (deg_v-1) * order.x + i + 1];
    vi_1 = vi_1 + vn * uv.y * points[baseid + (deg_v  ) * order.x + i + 1];
    vi = (1.0f - uv.y) * vi_0 + uv.y * vi_1;

    v0 = v1;
    v1 = vi;
    
    un = un * uv.x;
    bcu = bcu * (float)(deg_u - i) / (float)(i);
    u0 = (u0 + un * bcu * v0) * (1.0f - uv.x);
    u1 = (u1 + un * bcu * v1) * (1.0f - uv.x);
  }
  
  /********************************* 7. horner step for last column ****************************************/
  bcv = 1.0f; 
  vn = 1.0f; 
  vi_0 = points[baseid +           deg_u      ] * (1.0f - uv.y); 
  vi_1 = points[baseid + order.x + deg_u      ] * (1.0f - uv.y);
  
  for (i = 1; i <= deg_v-2; ++i) 
  { 
    vn = vn * uv.y; 
    bcv = bcv * (float)(deg_v-i) / (float)(i); 
    vi_0 = (vi_0 + vn * bcv * points[baseid + (i  ) * order.x + deg_u]) * (1.0f - uv.y);
    vi_1 = (vi_1 + vn * bcv * points[baseid + (i+1) * order.x + deg_u]) * (1.0f - uv.y);
   }
  
  vi_0 = vi_0 + vn * uv.y * points[baseid + (deg_v-1) * order.x + deg_u];
  vi_1 = vi_1 + vn * uv.y * points[baseid + (deg_v  ) * order.x + deg_u];
  vi = (1.0f - uv.y) * vi_0 + uv.y * vi_1;

  /******************************* 8. final interpolation ***************************************************/
  u0 = u0 + un * uv.x * v1;
  u1 = u1 + un * uv.x * vi;
  
  /* transform to euclidian space */
  du   = deg_u * ((u0.w * u1.w) / (point.w * point.w)) * ((u1 / u1.w) - (u0 / u0.w));
  point = point/point.w;
  p    = point;
}

#endif // LIB_GPUCAST_HORNER_SURFACE_H