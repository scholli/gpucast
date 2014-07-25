vec2 
env_long_lat(in vec3 v) 
{ 
  float invpi = 1.0f / 3.1415f;

  vec2 a_xz = normalize(v.xz); 
  vec2 a_yz = normalize(v.yz); 
 
  return vec2(0.5 * (1.0 + invpi * atan(a_xz.x, -a_xz.y)), 
              acos(-v.y) * invpi); 
} 

/****************************************************
* phong illumination including fresnel term and envmap
****************************************************/
vec4 shade_phong_fresnel ( in vec4        p_world,
                           in vec3        n_world,
                           in vec4        light0,
                           in vec3        matambient,
                           in vec3        matdiffuse,
                           in vec3        matspecular,
                           in float       shininess,
                           in float       opacity,
                           in bool        spheremapping,
                           in sampler2D   spheremap,
                           in bool        diffusemapping,
                           in sampler2D   diffusemap )
{
  vec4 result = vec4(0.0);

  p_world.w = 1.0;

  /* for all light sources */
  vec4 color_non_reflective = vec4(0.0);
  vec3 V = normalize(-p_world.xyz); // point to viewer
  vec3 R = vec3(0.0);               // reflected lightray
  vec3 L = vec3(0.0);               // point to lightsource
  
  /* correct normal, if backfacing */
  if (dot(V, n_world) < 0.0)
  {
    n_world *= -1.0;
  }
  vec3 N = normalize(n_world);

  const vec4 lightcolor0 = vec4(1.0);

  L = normalize(light0.xyz - p_world.xyz);
  R = reflect(-L, n_world);

  vec4 ambient  = vec4(matambient.xyz,  1.0)  * lightcolor0;
  vec4 diffuse  = vec4(matdiffuse.xyz,  1.0)  * lightcolor0 * max(0.0, dot(n_world, L));
  vec4 specular = vec4(matspecular.xyz, 1.0)  * lightcolor0 * pow(max(0.0, dot(R,V)), 20);

  color_non_reflective += vec4(ambient.xyz,  opacity);
  color_non_reflective += vec4(diffuse.xyz,  opacity);
  color_non_reflective += vec4(specular.xyz, opacity);

  vec2 spherecoords_N = env_long_lat(N);
  vec2 spherecoords_R = env_long_lat(R);

  if ( diffusemapping )
  {
    vec4 diffuse_in     = texture(diffusemap, spherecoords_N);
    vec4 color_diffuse  = vec4(matdiffuse, opacity) * diffuse_in;
    result              = color_diffuse;
  } else {
    result              = ambient;
    result             += diffuse;
  }

  /* spheremap for reflection */
  if ( spheremapping )
  {
    float R_phi    = shininess + (1.0f - shininess) * pow(1.0 - dot(n_world, V), 5);

    vec4 color_reflective = R_phi * vec4(matspecular, opacity) * texture(spheremap, spherecoords_R);
    result += color_reflective;
  } else {
    result += specular;
  }
  return result;
}

