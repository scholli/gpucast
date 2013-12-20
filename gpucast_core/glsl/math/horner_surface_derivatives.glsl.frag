/*******************************************************************************
 * Evaluate Surface using modificated horner algorithm in Bernstein basis
 * points assumed in homogenous coordinates! :  p = [wx wy wz w]
*******************************************************************************/
void 
evaluateSurface(in samplerBuffer  data,
                in int            index,
                in int            orderU,
                in int            orderV,
                in vec2           uv,
                out vec4          p, 
                out vec4          du, 
                out vec4          dv) 
{
  p  = vec4(0.0);
  du = vec4(0.0);
  dv = vec4(0.0);

  float bcu = 1.0;
  float un  = 1.0;

  int deg_u = orderU - 1;
  int deg_v = orderV - 1;

  vec4 u0_0 = texelFetchBuffer(data, index              ) * (1.0 - uv[0]);
  vec4 u0_1 = texelFetchBuffer(data, index + 1          ) * (1.0 - uv[0]);
  vec4 u1_0 = texelFetchBuffer(data, index + orderU    )  * (1.0 - uv[0]);
  vec4 u1_1 = texelFetchBuffer(data, index + orderU + 1)  * (1.0 - uv[0]);

  /**************************************** 1. step : horner for first 2 rows *********************************/
  int i;
  for (i = 1; i <= deg_u - 2; ++i) {
    un = un * uv[0];
    bcu = bcu * (float(deg_u - i) / float(i));

    u0_0 = (u0_0 + un * bcu * texelFetchBuffer(data, index + i    ))           * (1.0 - uv[0]);
    u0_1 = (u0_1 + un * bcu * texelFetchBuffer(data, index + i + 1))           * (1.0 - uv[0]);

    u1_0 = (u1_0 + un * bcu * texelFetchBuffer(data, index + orderU + i    )) * (1.0 - uv[0]);
    u1_1 = (u1_1 + un * bcu * texelFetchBuffer(data, index + orderU + i + 1)) * (1.0 - uv[0]);
  }

  u0_0 += un * uv[0] * texelFetchBuffer(data, index          + deg_u - 1);
  u0_1 += un * uv[0] * texelFetchBuffer(data, index          + deg_u    );
  u1_0 += un * uv[0] * texelFetchBuffer(data, index + orderU + deg_u - 1);
  u1_1 += un * uv[0] * texelFetchBuffer(data, index + orderU + deg_u    ); 
  
  /* point in first and second row */
  vec4 u0 = (1.0 - uv[0]) * u0_0 + uv[0] * u0_1;
  vec4 u1 = (1.0 - uv[0]) * u1_0 + uv[0] * u1_1; 

  /**************************************** 2. step : inner loop for rows 3 to order - 1 ***********************/
  float bcv = 1.0;
  float vn = 1.0;

  vec4 v0 = u0 * (1.0 - uv[1]);
  vec4 v1 = u1 * (1.0 - uv[1]);

  vec4 ui, ui_0, ui_1;
  for (i = 1; i <= deg_v - 2; ++i) 
  {
    bcu = 1.0;
    un = 1.0;
    ui_0 = texelFetchBuffer(data, index + (i+1) * orderU)     * (1.0 - uv[0]);
    ui_1 = texelFetchBuffer(data, index + (i+1) * orderU + 1) * (1.0 - uv[0]);

    int j;
    for (j = 1; j <= deg_u-2; ++j) 
    {
      un = un * uv[0];
      bcu = bcu * (float(deg_u - j) / float(j));
      ui_0 = (ui_0 + un * bcu * texelFetchBuffer(data, index + (i+1) * orderU + j    )) * (1.0 - uv[0]);
      ui_1 = (ui_1 + un * bcu * texelFetchBuffer(data, index + (i+1) * orderU + j + 1)) * (1.0 - uv[0]);
    }
    ui_0 = ui_0 + un * uv[0] * texelFetchBuffer(data, index + (i+1) * orderU + deg_u - 1);
    ui_1 = ui_1 + un * uv[0] * texelFetchBuffer(data, index + (i+1) * orderU + deg_u    );
    ui = (1.0 - uv[0]) * ui_0 + uv[0] * ui_1;

    u0 = u1;
    u1 = ui;
    
    vn = vn * uv[1];
    bcv = bcv * (float(deg_v - i) / float(i));
    v0 = (v0 + vn * bcv * u0) * (1.0 - uv[1]);
    v1 = (v1 + vn * bcv * u1) * (1.0 - uv[1]);
  }
  
  /**************************************** 3. step : horner scheme for last row *******************************/
  bcu = 1.0; 
  un = 1.0; 
  ui_0 = texelFetchBuffer(data, index + deg_v * orderU    ) * (1.0 - uv[0]); 
  ui_1 = texelFetchBuffer(data, index + deg_v * orderU + 1) * (1.0 - uv[0]);

  for (i = 1; i <= deg_u-2; ++i) 
  { 
    un = un * uv[0]; 
    bcu = bcu * (float(deg_u-i) / float(i)); 
    ui_0 = (ui_0 + un * bcu * texelFetchBuffer(data, index + deg_v * orderU + i    )) * (1.0 - uv[0]);
    ui_1 = (ui_1 + un * bcu * texelFetchBuffer(data, index + deg_v * orderU + i + 1)) * (1.0 - uv[0]);
  }

  ui_0 = ui_0 + un * uv[0] * texelFetchBuffer(data, index + deg_v * orderU + deg_u - 1);
  ui_1 = ui_1 + un * uv[0] * texelFetchBuffer(data, index + deg_v * orderU + deg_u    );
  ui = (1.0 - uv[0]) * ui_0 + uv[0] * ui_1;
  
  /**************************************** 4. step : final interpolation over v ********************************/
  v0 += vn * uv[1] * u1;
  v1 += vn * uv[1] * ui;

  p = (1.0 - uv[1]) * v0 + uv[1] * v1;

  /* transform to euclidian space */
  dv = (orderV - 1) * ((v0[3] * v1[3]) / (p[3] * p[3])) * ((v1 / v1[3]) - (v0 / v0[3]));

  /************************************ 5.step : dartial derivative over u ***********************************/
  bcv = 1.0;
  vn = 1.0;

  vec4 v0_0 = texelFetchBuffer(data, index              ) * (1.0 - uv[1]);
  vec4 v0_1 = texelFetchBuffer(data, index + orderU    ) * (1.0 - uv[1]);
  vec4 v1_0 = texelFetchBuffer(data, index + 1          ) * (1.0 - uv[1]);
  vec4 v1_1 = texelFetchBuffer(data, index + orderU + 1) * (1.0 - uv[1]);

  for (i = 1; i <= deg_v - 2; ++i) 
  {
    vn = vn * uv[1];
    bcv = bcv * (float(deg_v - i) / float(i));

    v0_0 = (v0_0 + vn * bcv * texelFetchBuffer(data, index + (i  ) * orderU    )) * (1.0 - uv[1]);
    v0_1 = (v0_1 + vn * bcv * texelFetchBuffer(data, index + (i+1) * orderU    )) * (1.0 - uv[1]);

    v1_0 = (v1_0 + vn * bcv * texelFetchBuffer(data, index + (i  ) * orderU + 1)) * (1.0 - uv[1]);
    v1_1 = (v1_1 + vn * bcv * texelFetchBuffer(data, index + (i+1) * orderU + 1)) * (1.0 - uv[1]);
  }

  v0_0 = v0_0 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v-1) * orderU    );
  v0_1 = v0_1 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v  ) * orderU    );
  v1_0 = v1_0 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v-1) * orderU + 1);
  v1_1 = v1_1 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v  ) * orderU + 1); 
  
  /* point in first and second row */
  v0 = (1.0 - uv[1]) * v0_0 + uv[1] * v0_1;
  v1 = (1.0 - uv[1]) * v1_0 + uv[1] * v1_1; 

  /*********************************** 6. step : for all columns *******************************************/
  bcu = 1.0;
  un = 1.0;

  u0 = v0 * (1.0 - uv[0]);
  u1 = v1 * (1.0 - uv[0]);

  vec4 vi_0, vi_1, vi;
  for (i = 1; i <= deg_u - 2; ++i) 
  {
    bcv = 1.0;
    vn = 1.0;
    vi_0 = texelFetchBuffer(data, index +           i + 1) * (1.0 - uv[1]);
    vi_1 = texelFetchBuffer(data, index + orderU + i + 1) * (1.0 - uv[1]);

    int j;
    for (j = 1; j <= deg_v-2; ++j) 
    {
      vn = vn * uv[1];
      bcv = bcv * (float(deg_v - j) / float(j));
      vi_0 = (vi_0 + vn * bcv * texelFetchBuffer(data, index + (j  ) * orderU + i + 1)) * (1.0 - uv[1]);
      vi_1 = (vi_1 + vn * bcv * texelFetchBuffer(data, index + (j+1) * orderU + i + 1)) * (1.0 - uv[1]);
    }
    vi_0 = vi_0 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v-1) * orderU + i + 1);
    vi_1 = vi_1 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v  ) * orderU + i + 1);
    vi = (1.0 - uv[1]) * vi_0 + uv[1] * vi_1;

    v0 = v1;
    v1 = vi;
    
    un = un * uv[0];
    bcu = bcu * (float(deg_u - i) / float(i));
    u0 = (u0 + un * bcu * v0) * (1.0 - uv[0]);
    u1 = (u1 + un * bcu * v1) * (1.0 - uv[0]);
  }
  
  /********************************* 7. horner step for last column ****************************************/
  bcv = 1.0; 
  vn = 1.0; 
  vi_0 = texelFetchBuffer(data, index +           deg_u      ) * (1.0 - uv[1]); 
  vi_1 = texelFetchBuffer(data, index + orderU + deg_u      ) * (1.0 - uv[1]);
  
  for (i = 1; i <= deg_v-2; ++i) 
  { 
    vn = vn * uv[1]; 
    bcv = bcv * (float(deg_v-i) / float(i)); 
    vi_0 = (vi_0 + vn * bcv * texelFetchBuffer(data, index + (i  ) * orderU + deg_u)) * (1.0 - uv[1]);
    vi_1 = (vi_1 + vn * bcv * texelFetchBuffer(data, index + (i+1) * orderU + deg_u)) * (1.0 - uv[1]);
   }
  
  vi_0 = vi_0 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v-1) * orderU + deg_u);
  vi_1 = vi_1 + vn * uv[1] * texelFetchBuffer(data, index + (deg_v  ) * orderU + deg_u);
  vi = (1.0 - uv[1]) * vi_0 + uv[1] * vi_1;

  /******************************* 8. final interpolation ***************************************************/
  u0 += un * uv[0] * v1;
  u1 += un * uv[0] * vi;
  
  /* transform to euclidian space */
  du = deg_u * ((u0[3] * u1[3]) / (p[3] * p[3])) * ((u1 / u1[3]) - (u0 / u0[3]));
  p = p/p[3];   
}

