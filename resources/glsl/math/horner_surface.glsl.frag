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
                out vec4          p) 
{
  p = vec4(0.0);

  float bcu = 1.0;
  float un = 1.0;
  int deg_u = orderU - 1;
  int deg_v = orderV - 1;

  vec4 u0_0 = texelFetchBuffer(data, index              ) * (1.0 - uv[0]);
  vec4 u0_1 = texelFetchBuffer(data, index + 1          ) * (1.0 - uv[0]);
  vec4 u1_0 = texelFetchBuffer(data, index + orderU    ) * (1.0 - uv[0]);
  vec4 u1_1 = texelFetchBuffer(data, index + orderU + 1) * (1.0 - uv[0]);

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

  u0_0 += un * uv[0] * texelFetchBuffer(data, index           + deg_u - 1);
  u0_1 += un * uv[0] * texelFetchBuffer(data, index           + deg_u    );
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
  for (i = 1; i <= deg_v - 2; ++i) {
    bcu = 1.0;
    un = 1.0;
    ui_0 = texelFetchBuffer(data, index + (i+1) * orderU)     * (1.0 - uv[0]);
    ui_1 = texelFetchBuffer(data, index + (i+1) * orderU + 1) * (1.0 - uv[0]);

    int j;
    for (j = 1; j <= deg_u-2; ++j) {
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

  for (i = 1; i <= deg_u-2; ++i) { 
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
  p = p/p[3];   
}


