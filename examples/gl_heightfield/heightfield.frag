
#version 330 compatibility
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D colortex;
uniform sampler2D normaltex;
uniform sampler2D bumptex;
uniform vec4 lightpos;
uniform int mode;

uniform int texture_width;
uniform int texture_height;

uniform mat4 normalmatrix;

in vec4 frag_texcoord;
in vec4 frag_position;
in vec4 frag_normal;

out vec4 out_color;

void main(void)
{
  // partial derivatives
  vec3 q0  = dFdx(frag_position.xyz);
	vec3 q1  = dFdy(frag_position.xyz);
	vec2 st0 = dFdx(frag_texcoord.st);
	vec2 st1 = dFdy(frag_texcoord.st);

  // view vector and surface normal
  vec3 V = normalize(-frag_position.xyz);
  vec3 FN = normalize(frag_normal.xyz);
  
  // transform texture coordinates into world space
	vec3 S = normalize( q0 * st1.t - q1 * st0.t);
	vec3 T = normalize(-q0 * st1.s + q1 * st0.s);
	mat3 M = mat3(-T, -S, FN);
	
	// ray cast through bump map
  vec3 ray_world  = normalize(frag_position.xyz);
  
  // project ray into texture space
  float dx        = dot(ray_world, S);
  float dy        = dot(ray_world, T);
  
  vec2 ray_tex = vec2(dx, dy);
  
  bool  hit       = false;
  
  vec2  stepwidth = vec2(1.0/texture_width, 1.0/texture_width);
  int   iters     = 0;
  float depth     = 1.0;
  vec2  current_texcoord = frag_texcoord.st;
  float scale     = 0.6;
	
  while (hit != true && iters < texture_width)
  {
    // get height for current sample
    vec4 sample = texture2D(bumptex, current_texcoord);
    
    // if depth in heightfield is greater than current ray depth
    if (sample.x > depth) {
      hit = true;
    }
    
    // if ray casting steps out of texture
    if (current_texcoord.x > 1.0 || 
        current_texcoord.y > 1.0 || 
        current_texcoord.x < 0.0 || 
        current_texcoord.y < 0.0)
    {
      discard;
    }
    
    // do iteration
    current_texcoord += stepwidth * ray_tex;
    depth -= ((1.0 - dot(FN, ray_world)) / texture_width) / scale;
    ++iters;
  }
	
  // lighting
  vec3 N = normalize(M * (vec3(texture2D(normaltex, current_texcoord)) - vec3(0.5, 0.5, 0.5)));
  vec3 L = normalize(lightpos.xyz - frag_position.xyz);  
  vec3 R = normalize(reflect(-L, N));         
 
  vec4 c  = texture2D(colortex, current_texcoord);
  vec4 h  = texture2D(bumptex, current_texcoord);
  
  if (hit) {
    out_color = c * vec4(dot(N,L) + pow(max(dot(R,N), 0.0), 150.0));
  } else {
    discard;
  }
};


