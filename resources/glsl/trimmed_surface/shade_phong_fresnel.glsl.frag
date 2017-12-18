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
                           in vec3        view_direction,
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
  vec3 V = view_direction;          // vector to viewer
  vec3 R = vec3(0.0);               // reflected lightray
  vec3 L = vec3(0.0);               // point to lightsource
  
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

  result += color_non_reflective;

  vec2 spherecoords_N = env_long_lat(N);
  vec2 spherecoords_R = env_long_lat(R);

  /* spheremap for reflection */
  if ( spheremapping )
  {
  #if 0
    float R_phi    = shininess + (1.0f - shininess) * pow(1.0 - dot(n_world, V), 5);
    vec4 color_reflective = R_phi * vec4(matspecular, opacity) * texture(spheremap, spherecoords_R);
    result += color_reflective;
  #else
    float R_phi    = shininess + (1.0f - shininess) * pow(1.0 - dot(n_world, V), 5);

    vec4 cam_world = vec4(0.0, 0.0, 0.0, 1.0) * gpucast_view_inverse_matrix;
    vec4 view_dir_world = cam_world - p_world;
    view_dir_world.w =0.0;
    view_dir_world = normalize(view_dir_world);
    vec3 reflected_view_dir = reflect(view_dir_world.xyz, n_world);
    
    float tu = asin(reflected_view_dir.x)/ 3.14159265359  + 0.5;
    float tv = asin(reflected_view_dir.y)/ 3.14159265359  + 0.5;

  	vec2 texcoord = vec2(tu, tv);
    vec4 color_reflective = clamp(R_phi, 0.0, 1.0) * vec4(matspecular, opacity) * texture(spheremap, texcoord);
    result += color_reflective;
    
  #endif
  } else {
    result += specular;
  }
  return result;
}

