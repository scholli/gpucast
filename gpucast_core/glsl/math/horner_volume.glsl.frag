/*******************************************************************************
 * Evaluate Volume using modificated horner algorithm in Bernstein basis
 * index assumed in homogenous coordinates! :  p = [wx wy wz w]
*******************************************************************************/
void 
evaluateVolume (in samplerBuffer  data,
                in int            index,
                in int            order_u,
                in int            order_v,
                in int            order_w,
                in float          u, 
                in float          v, 
                in float          w, 
                out vec4          point, 
                out vec4          du, 
                out vec4          dv,
                out vec4          dw) 
{
  point = vec4(0.0);
  du    = vec4(0.0);
  dv    = vec4(0.0);
  dw    = vec4(0.0);

  // helper like binomial coefficients and t^n
  float one_minus_u  = float(1.0) - u;
  float one_minus_v  = float(1.0) - v;
  float one_minus_w  = float(1.0) - w;

  float bcw0          = 1;
  float wn0           = 1;
  float bcw1          = 1;
  float wn1           = 1;
  
  vec4 p000 = vec4(0.0);
  vec4 p001 = vec4(0.0);
  vec4 p010 = vec4(0.0);
  vec4 p011 = vec4(0.0);
  vec4 p100 = vec4(0.0);
  vec4 p101 = vec4(0.0);
  vec4 p110 = vec4(0.0);
  vec4 p111 = vec4(0.0);

  // evaluate using horner scheme
  for (int k = 0; k != order_w; ++k)
  {
    vec4 pv = vec4(0.0);

    vec4 pv00 = vec4(0.0);
    vec4 pv01 = vec4(0.0);
    vec4 pv10 = vec4(0.0);
    vec4 pv11 = vec4(0.0);

    float bcv0 = 1;
    float bcv1 = 1;
    float vn0  = 1;
    float vn1  = 1;

    for (int j = 0; j != order_v; ++j)
    {
      vec4 pu = vec4(0.0); // temporaryly
      vec4 pu0;
      vec4 pu1;
      float bcu = 1;
      float un  = 1;
      
      // first interpolation (1-u)^n    
      pu0 = texelFetchBuffer(data, index +     j * order_u + k * order_u * order_v ) * one_minus_u;      
      pu1 = texelFetchBuffer(data, index + 1 + j * order_u + k * order_u * order_v ) * one_minus_u;

      for (int i = 1; i < order_u-2; ++i)
      {
        // follow regular horner scheme
        un  *= u;
        bcu *= float(order_u - 1 - i) / float(i);
        pu0 = (pu0 + un * bcu * texelFetchBuffer(data, index +     i + j * order_u + k * order_u * order_v )) * one_minus_u; 
        pu1 = (pu1 + un * bcu * texelFetchBuffer(data, index + 1 + i + j * order_u + k * order_u * order_v )) * one_minus_u; 
      }

      // last interpolation u^n
      pu0 += un * u * texelFetchBuffer(data, index + order_u-2 + j * order_u + k * order_u * order_v );
      pu1 += un * u * texelFetchBuffer(data, index + order_u-1 + j * order_u + k * order_u * order_v );

      //pu = (1-u) * pu0 + u * pu1;
      pu = mix(pu0, pu1, u);

      if (j == 0) { // first interpolation (1-v)^n    
        pv00 = pu0 * one_minus_v;
        pv10 = pu1 * one_minus_v;
      }

      if (j == 1) {
        pv01 = pu0 * one_minus_v;
        pv11 = pu1 * one_minus_v;
      }

      if (j == 1 && order_v > 3) {
        vn0  *= v;
        bcv0 *= float(order_v - 1 - j) / float(j);
        pv00  = (pv00 + vn0 * bcv0 * pu0) * one_minus_v; 
        pv10  = (pv10 + vn0 * bcv0 * pu1) * one_minus_v; 
      }

      if (j == order_v - 2) {
        pv00 += vn0 * v * pu0;
        pv10 += vn0 * v * pu1;
      }

      if (j == order_v - 2 && order_v > 3) {
        vn1  *= v;
        bcv1 *= float(order_v - j) / float(j-1);
        pv01  = (pv01 + vn1 * bcv1 * pu0) * one_minus_v; 
        pv11  = (pv11 + vn1 * bcv1 * pu1) * one_minus_v; 
      }

      if (j == order_v - 1) {
        pv01 += vn1 * v * pu0;
        pv11 += vn1 * v * pu1;
      }

      if ( j > 1 && j < order_v - 2 && order_v > 3) {
        vn0  *= v;
        vn1  *= v;
        bcv0 *= float(order_v - 1 - j) / float(j);
        bcv1 *= float(order_v     - j) / float(j-1);
        
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

    if (k == 1 && order_w > 3) {
      wn0  *= w;
      bcw0 *= float(order_w - 1 - k) / float(k);
      p000  = (p000 + wn0 * bcw0 * pv00) * one_minus_w; 
      p100  = (p100 + wn0 * bcw0 * pv10) * one_minus_w; 
      p010  = (p010 + wn0 * bcw0 * pv01) * one_minus_w; 
      p110  = (p110 + wn0 * bcw0 * pv11) * one_minus_w; 
    }

    if (k == order_w - 2) {
      p000 += wn0 * w * pv00;
      p100 += wn0 * w * pv10;
      p010 += wn0 * w * pv01;
      p110 += wn0 * w * pv11;
    }

    if (k == order_w - 2 && order_w > 3) {
      wn1  *= w;
      bcw1 *= float(order_w - k) / float(k-1);

      p001  = (p001 + wn1 * bcw1 * pv00) * one_minus_w; 
      p101  = (p101 + wn1 * bcw1 * pv10) * one_minus_w; 
      p011  = (p011 + wn1 * bcw1 * pv01) * one_minus_w; 
      p111  = (p111 + wn1 * bcw1 * pv11) * one_minus_w; 
    }

    if (k == order_w - 1) {
      p001 += wn1 * w * pv00;
      p101 += wn1 * w * pv10;
      p011 += wn1 * w * pv01;
      p111 += wn1 * w * pv11;
    }

    if ( k > 1 && k < order_w - 2 && order_w > 3) {
      wn0  *= w;
      wn1  *= w;
      bcw0 *= float(order_w - 1 - k) / float(k);
      bcw1 *= float(order_w     - k) / float(k-1);
      
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
  vec4 vw00 = mix(p000, p100, u);
  vec4 vw10 = mix(p010, p110, u);
  vec4 vw01 = mix(p001, p101, u);
  vec4 vw11 = mix(p011, p111, u);

  // evaluate for v leaving a linear patch dependending on u,w
  vec4 uw00 = mix(p000, p010, v);
  vec4 uw10 = mix(p100, p110, v);
  vec4 uw01 = mix(p001, p011, v);
  vec4 uw11 = mix(p101, p111, v);

  // evaluating v,w plane for v resulting in last linear interpolation in w -> to compute firs  t partial derivative in w
  vec4 w0 = mix(vw00, vw10, v);
  vec4 w1 = mix(vw01, vw11, v);

  // evaluating v,w plane for w resulting in last linear interpolation in v -> to compute first partial derivative in v
  vec4 v0 = mix(vw00, vw01, w);
  vec4 v1 = mix(vw10, vw11, w);

  // evaluating v,w plane for w resulting in last linear interpolation in v -> to compute first partial derivative in v
  vec4 u0 = mix(uw00, uw01, w);
  vec4 u1 = mix(uw10, uw11, w);

  // last interpolation and back projection to euclidian space
  point = mix(w0, w1, w);

  point = euclidian_space(point);

  // M.S. Floater '91 :
  //
  //             w[0]{n-1}(t) * w[1]{n-1}(t)
  // P'(t) = n * --------------------------- * P[1]{n-1}(t) - P[0]{n-1}(t)
  //                     w[0]{n})^2
  //
  // 1. recalculate overwritten helping point P[0, n-1]
  // 2. project P[0, n-1] and P[1, n-1] into plane w=1
  // 3. use formula above to find out the correct length of P'(t)

  du = (order_u - float(1)) * ((u0[3] * u1[3]) / (point[3] * point[3])) * (euclidian_space(u1) - euclidian_space(u0));
  dv = (order_v - float(1)) * ((v0[3] * v1[3]) / (point[3] * point[3])) * (euclidian_space(v1) - euclidian_space(v0));
  dw = (order_w - float(1)) * ((w0[3] * w1[3]) / (point[3] * point[3])) * (euclidian_space(w1) - euclidian_space(w0));
}



/*******************************************************************************
 * Evaluate Volume using modificated horner algorithm in Bernstein basis
 * index assumed in homogenous coordinates! :  p = [wx wy wz w]
*******************************************************************************/
void
evaluateVolume (in samplerBuffer  data,
                in int            index,
                in int            order_u,
                in int            order_v,
                in int            order_w,
                in float          u, 
                in float          v, 
                in float          w, 
                out vec4          point ) 
{
  point = vec4(0.0);

  // helper like binomial coefficients and t^n
  float one_minus_u  = float(1.0) - u;
  float one_minus_v  = float(1.0) - v;
  float one_minus_w  = float(1.0) - w;

  float bcw          = 1;
  float wn           = 1;

  // evaluate using horner scheme
  for (int k = 0; k != order_w; ++k)
  {
    vec4 pv = vec4(0.0);
    float bcv = 1;
    float vn  = 1;

    for (int j = 0; j != order_v; ++j)
    {
      vec4 pu = vec4(0.0);
      float bcu = 1;
      float un  = 1;

      for (int i = 0; i < order_u; ++i)
      {
        if (i == 0) 
        { // first interpolation (1-u)^n    
            pu = texelFetchBuffer(data, index + j * order_u + k * order_u * order_v) * one_minus_u;
        } else {
          if (i == order_u - 1) { // last interpolation u^n
            pu += un * u * texelFetchBuffer(data, index + i + j * order_u + k * order_u * order_v );
          } else {  // else follow regular horner scheme
            un  *= u;
            bcu *= float(order_u - i) / float(i);
            pu = (pu + un * bcu * texelFetchBuffer(data, index + i + j * order_u + k * order_u * order_v )) * one_minus_u;
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
          bcv *= float(order_v - j) / float(j);
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
        bcw *= float(order_w - k) / float(k);
        point= (point + wn * bcw * pv) * one_minus_w;
      }
    }
  }

  point = euclidian_space(point);
}

