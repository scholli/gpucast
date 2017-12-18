// Hash without Sine
// Creative Commons Attribution-ShareAlike 4.0 International Public License
// Created by David Hoskins.

// https://www.shadertoy.com/view/4djSRW
// Trying to find a Hash function that is the same on ALL systens
// and doesn't rely on trigonometry functions that change accuracy 
// depending on GPU. 
// New one on the left, sine function on the right.
// It appears to be the same speed, but I suppose that depends.

// * Note. It still goes wrong eventually!
// * Try full-screen paused to see details.


#define ITERATIONS 4


// *** Change these to suit your range of random numbers..

// *** Use this for integer stepped ranges, ie Value-Noise/Perlin noise functions.
#define HASHSCALE1 .1031
#define HASHSCALE3 vec3(.1031, .1030, .0973)
#define HASHSCALE4 vec4(.1031, .1030, .0973, .1099)

// For smaller input rangers like audio tick or 0-1 UVs use these...
//#define HASHSCALE1 443.8975
//#define HASHSCALE3 vec3(443.897, 441.423, 437.195)
//#define HASHSCALE4 vec3(443.897, 441.423, 437.195, 444.129)



//----------------------------------------------------------------------------------------
//  1 out, 1 in...
float hash11(float p)
{
  vec3 p3 = fract(vec3(p) * HASHSCALE1);
  p3 += dot(p3, p3.yzx + 19.19);
  return fract((p3.x + p3.y) * p3.z);
}

//----------------------------------------------------------------------------------------
//  1 out, 2 in...
float hash12(vec2 p)
{
  vec3 p3 = fract(vec3(p.xyx) * HASHSCALE1);
  p3 += dot(p3, p3.yzx + 19.19);
  return fract((p3.x + p3.y) * p3.z);
}

///////////////////////////////////////////////////////////////////////////////
vec4 depth_to_object_space(in float depth, in vec2 uv, in mat4 mvp_inverse) 
{
  vec4 ndc = vec4(1.0);
  
  ndc.z = (depth - 0.5) * 2.0;
  ndc.x = (uv.x - 0.5) * 2.0;
  ndc.y = (uv.y - 0.5) * 2.0;

  vec4 oc = mvp_inverse * ndc;
  oc = oc / oc.w;

  return oc;
}

///////////////////////////////////////////////////////////////////////////////
vec4 depth_to_object_space(in sampler2D depth_buffer, in vec2 uv, in mat4 mvp_inverse)
{
  float depth = texture(depth_buffer, uv).x;
  return depth_to_object_space(depth, uv, mvp_inverse);
}

///////////////////////////////////////////////////////////////////////////////
vec4 estimate_object_space_normal(in sampler2D depth_buffer, in vec2 uv, in vec2 resolution, in mat4 mvp_inverse, in mat4 modelview)
{
  vec2 ires = vec2(1.0) / resolution;

  float depth = texture(depth_buffer, uv).x;

  if (depth == 1.0) {
    return vec4(0.0);
  }

  vec4 p   = depth_to_object_space(depth, uv, mvp_inverse);

  vec4 px0 = depth_to_object_space(depth_buffer, uv + vec2(ires.x, 0.0), mvp_inverse);
  vec4 px1 = depth_to_object_space(depth_buffer, uv - vec2(ires.x, 0.0), mvp_inverse);
  vec4 py0 = depth_to_object_space(depth_buffer, uv + vec2(0.0,  ires.y), mvp_inverse);
  vec4 py1 = depth_to_object_space(depth_buffer, uv - vec2(0.0,  ires.y), mvp_inverse);

  vec3 dx0 = px0.xyz - p.xyz;
  vec3 dx1 = px1.xyz - p.xyz;
  vec3 dy0 = py0.xyz - p.xyz;
  vec3 dy1 = py1.xyz - p.xyz;

  vec3 dx = 3 * length(dx0) < length(dx1) ? dx0 : dx1;
  vec3 dy = 3 * length(dy0) < length(dy1) ? dy0 : dy1;
  
  vec3 recons_normal = normalize(cross(normalize(dx), normalize(dy)));
  vec4 norm_vs = modelview * vec4(recons_normal, 0.0);

  if (norm_vs.z > 0.0) {
    return vec4(recons_normal.xyz, 1.0);
  }
  else {
    return vec4(-recons_normal.xyz, -1.0);
  }
  
}

///////////////////////////////////////////////////////////////////////////////
vec4 compute_ssao(in sampler2D depth_buffer, in sampler2D randomtex, in float screen_distance, in float radius, in int samples, in vec2 uv, in vec2 resolution, in float nearclip, in float farclip, in mat4 mvp_inverse, in mat4 modelview)
{
  float depth = texture(depth_buffer, uv).x;

  if (depth == 1.0) {
    return vec4(1.0);
  }

  vec4 p0 = depth_to_object_space(depth_buffer, uv, mvp_inverse);
  vec4 n0 = estimate_object_space_normal(depth_buffer, uv, resolution, mvp_inverse, modelview);

  vec2 ires = vec2(1.0) / resolution;

  float illumination = 0.0;
  float max_weight = 0.0;

  vec2 seed = resolution* uv.xy;

  for (int i = 0; i != samples; ++i) 
  {
    float u = hash12(seed);
    seed += u;
    float v = hash12(seed);
    seed -= v;
    vec2 random_offset_texcoord = 2.0 * (vec2(u, v) - vec2(0.5));

    vec2 offset = ires * screen_distance * random_offset_texcoord;

    float depth1 = texture(depth_buffer, uv + offset).x;
    vec4 p1 = depth_to_object_space(depth1, uv + offset, mvp_inverse);

    vec3 d1 = p1.xyz - p0.xyz;

    float sample_distance = length(d1);

    float is_in_radius = float(sample_distance < radius);
    float weighted_distance = is_in_radius * (1.0 - sample_distance / radius);
    float occlusion = weighted_distance * clamp(dot(normalize(d1), n0.xyz), 0.0, 1.0);
    
    const float t = 0.7;
    float weight = mix(weighted_distance, occlusion, t);

    max_weight += weight;

    if (occlusion > 0.0) {
      illumination += weight;
    }

  }

  float ambient_occlusion = clamp(illumination / max_weight, 0.0, 1.0);

  return vec4(ambient_occlusion, ambient_occlusion, ambient_occlusion, 1.0);
}
