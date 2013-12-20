/**********************************************************************
* generate ray defined by intersecting planes
***********************************************************************/
void 
raygen(in vec4    p_objectspace,
       in mat4    mv_inv,
       out vec3   n1, 
       out vec3   n2, 
       out float  d1, 
       out float  d2)
{
  vec4 rayorigin = vec4(mv_inv[3][0], 
			                  mv_inv[3][1], 
			                  mv_inv[3][2], 1.0);

  vec4 raydir = normalize(p_objectspace - rayorigin);

  if (abs(raydir[0]) > abs(raydir[1]) && abs(raydir[0]) > abs(raydir[2])) {
    n1 = vec3(raydir[1], -raydir[0], 0.0);
  } else {
    n1 = vec3(0.0, raydir[2], -raydir[1]);
  }

  n2 = cross(n1, raydir.xyz);
  d1 = dot(-n1, rayorigin.xyz);
  d2 = dot(-n2, rayorigin.xyz);
}


/////////////////////////////////////////////////////////////////////
void 
raygen (  in vec4 ray_origin,
          in vec4 ray_direction,
          out vec3 n1,
          out vec3 n2,
          out float d1,
          out float d2 )
{
  if ( abs(ray_direction[0]) > abs(ray_direction[1]) && 
       abs(ray_direction[0]) > abs(ray_direction[2])) 
  {
    n1 = vec3(ray_direction[1], -ray_direction[0], 0.0);
  } else {
    n1 = vec3(0.0, ray_direction[2], -ray_direction[1]);
  }

  n2 = cross(n1, ray_direction.xyz);

  n1 = normalize(n1);
  n2 = normalize(n2);

  d1 = dot(-n1, ray_origin.xyz);
  d2 = dot(-n2, ray_origin.xyz);
}

