/**********************************************************************
* newton iteration to find surface intersection
***********************************************************************/
bool
newton(inout vec2       uv,
       in float         epsilon,
       in int           max_steps,
       in samplerBuffer data,
       in int           index,
       in int           orderu,
       in int           orderv,
       in vec3          n1, 
       in vec3          n2,
       in float         d1, 
       in float         d2,
       out vec4         p, 
       out vec4         du, 
       out vec4         dv)
{
  p  = vec4(0.0);
  du = vec4(0.0);
  dv = vec4(0.0);

  vec2 Fuv = vec2(0.0);

  for (int i = 0; i < max_steps; ++i) 
  {
    evaluateSurface(data, index, orderu, orderv, uv, p, du, dv);

    Fuv       = vec2(dot(n1, p.xyz) + d1, dot(n2, p.xyz) + d2);

    vec2 Fu   = vec2(dot(n1, du.xyz), dot(n2, du.xyz));  
    vec2 Fv   = vec2(dot(n1, dv.xyz), dot(n2, dv.xyz));  

    mat2 J    = mat2(Fu, Fv); 

    float det = determinant(J);
    mat2 Jinv = adjoint(J) / det;

    uv = uv - Jinv * Fuv; 

    if (length(Fuv) < epsilon) {
      break;
    }

  } 

  bool uv_distance_too_large = length(Fuv) > epsilon;
  bool uv_out_of_domain = uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0;

  // return if convergence was reached
  return !uv_distance_too_large && !uv_out_of_domain;
}

