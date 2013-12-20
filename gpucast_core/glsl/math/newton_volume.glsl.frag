/**********************************************************************
* newton iteration to iterate to determine the parameter values for a sample point
***********************************************************************/
bool
newton_volume (in samplerBuffer data,
               in int           index_volume,
               in int           orderu,
               in int           orderv,
               in int           orderw,
               in vec3          in_uvw,
               out vec3         uvw,
               in vec4          target_point,
               out vec4         point,
               out vec4         du,
               out vec4         dv,
               out vec4         dw,
               in vec3          n1, 
               in vec3          n2,
               in vec3          n3,
               in float         d1, 
               in float         d2,
               in float         epsilon,
               in int           maxsteps
               )
{
  // init
  point = vec4(0.0);
  du    = vec4(0.0);
  dv    = vec4(0.0);
  dw    = vec4(0.0);

  // start iteration at in_uvw
  uvw = in_uvw;

  // set target point in volume
  float d3    = dot(-n3, target_point.xyz);

	for (int i = 1; i != maxsteps; ++i)
  {
    // evaluate volume at starting point
    evaluateVolume(data, index_volume, orderu, orderv, orderw, uvw.x, uvw.y, uvw.z, point, du, dv, dw);

		// compute distance to planes -> stepwidth
		vec3 Fuvw = vec3(dot(point.xyz, n1) + d1, dot(point.xyz, n2) + d2, dot(point.xyz, n3) + d3);
	
    // abort criteria I: approximate intersection reached
    if (length(Fuvw) < epsilon) {
		  break;
    }

		// map base vectors to planes
		vec3 Fu   = vec3( dot(n1, du.xyz), dot(n2, du.xyz), dot(n3, du.xyz));
		vec3 Fv   = vec3( dot(n1, dv.xyz), dot(n2, dv.xyz), dot(n3, dv.xyz));
		vec3 Fw   = vec3( dot(n1, dw.xyz), dot(n2, dw.xyz), dot(n3, dw.xyz));
	
		mat3 J    = mat3(Fu, Fv, Fw);

    // stop iteration if there's no solution 
    if ( determinant(J) == 0.0 ) 
    {
      break;
    }

    mat3 Jinv = inverse(J);
	
    // do step in parameter space
		uvw      -= Jinv * Fuvw;

    // clamp result to regular domain
    uvw       = clamp(uvw, vec3(0.0), vec3(1.0));
  }

  vec3 error = vec3(dot(point.xyz, n1) + d1, dot(point.xyz, n2) + d2, dot(point.xyz, n3) + d3);

  // success criteria -> point near sample & parameter in domain
	return ( length(error) <= epsilon );
}

