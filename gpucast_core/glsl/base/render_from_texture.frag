/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : render_from_texture.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* constants
********************************************************************************/

/********************************************************************************
* uniforms
********************************************************************************/
uniform sampler2D colorbuffer;

uniform int       width;
uniform int       height;
uniform int       enable;

uniform float     FXAA_SPAN_MAX = 8.0;
uniform float     FXAA_REDUCE_MUL = 1.0/8.0;

#define FxaaInt2 ivec2
#define FxaaFloat2 vec2
#define FxaaTexLod0(t, p) texture2DLod(t, p, 0.0)
#define FxaaTexOff(t, p, o, r) texture2DLodOffset(t, p, 0.0, o)

/********************************************************************************
* input
********************************************************************************/
in vec4 frag_texcoord;
in vec4 fxaa_pos;

/********************************************************************************
* output
********************************************************************************/
layout (location=0) out vec4 out_color;

/********************************************************************************
* functions
********************************************************************************/
vec3 FxaaPixelShader(
  vec4 posPos, // Output of FxaaVertexShader interpolated across screen.
  sampler2D tex, // Input texture.
  vec2 rcpFrame) // Constant {1.0/frameWidth, 1.0/frameHeight}.
{
/*---------------------------------------------------------*/
    #define FXAA_REDUCE_MIN   (1.0/128.0)
    //#define FXAA_REDUCE_MUL   (1.0/8.0)
    //#define FXAA_SPAN_MAX     8.0
/*---------------------------------------------------------*/
    vec3 rgbNW = textureLod(tex, posPos.zw, 0.0).xyz;
    vec3 rgbNE = textureLodOffset(tex, posPos.zw, 0.0, ivec2(1,0)).xyz;
    vec3 rgbSW = textureLodOffset(tex, posPos.zw, 0.0, ivec2(0,1)).xyz;
    vec3 rgbSE = textureLodOffset(tex, posPos.zw, 0.0, ivec2(1,1)).xyz;
    vec3 rgbM  = textureLod(tex, posPos.xy, 0.0).xyz;
/*---------------------------------------------------------*/
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
/*---------------------------------------------------------*/
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
/*---------------------------------------------------------*/
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
/*---------------------------------------------------------*/
    float dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),
        FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),
          max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
          dir * rcpDirMin)) * rcpFrame.xy;
/*--------------------------------------------------------*/
    vec3 rgbA = (1.0/2.0) * (
        textureLod(tex, posPos.xy + dir * (1.0/3.0 - 0.5), 0.0).xyz +
        textureLod(tex, posPos.xy + dir * (2.0/3.0 - 0.5), 0.0).xyz);
    vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (
        textureLod(tex, posPos.xy + dir * (0.0/3.0 - 0.5), 0.0).xyz +
        textureLod(tex, posPos.xy + dir * (3.0/3.0 - 0.5), 0.0).xyz);
    float lumaB = dot(rgbB, luma);
    if((lumaB < lumaMin) || (lumaB > lumaMax)) return rgbA;
    return rgbB; }

///////////////////////////////////////////////////////////////////////////////
vec4 PostFX(in sampler2D  tex, 
            in vec2       uv, 
            in vec4       pos, 
            in int        screen_width, 
            in int        screen_height)
{
  vec4 c = vec4(0.0);
  vec2 rcpFrame = vec2(1.0/screen_width, 1.0/screen_height);
  c.rgb = FxaaPixelShader(pos, tex, rcpFrame);
  //c.rgb = 1.0 - texture2D(tex, posPos.xy).rgb;
  c.a = 1.0;
  return c;
}


/********************************************************************************
* clean pass cleans index image texture
********************************************************************************/
void main(void)
{
  if ( enable > 0)
  {
    out_color = PostFX(colorbuffer, frag_texcoord.xy, fxaa_pos, width, height);
  } else {
    out_color = texture(colorbuffer, frag_texcoord.xy);
  }
}
