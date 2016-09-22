#version 330 compatibility

uniform sampler2D colortex;
uniform sampler2D normaltex;
uniform sampler2D bumptex;
uniform vec4 lightpos;
uniform int mode;

uniform mat4 normalmatrix;

in vec4 frag_texcoord;
in vec4 frag_position;
in vec4 frag_normal;

out vec4 out_color;

void main(void)
{
  float depth_bias = 0.4;

  // bump mapping
  vec3 V = normalize(-frag_position.xyz);
  vec3 FN = normalize(frag_normal.xyz);

  vec4 b = texture2D(bumptex,    frag_texcoord.xy);
  vec2 offset = vec2(V.x, -V.y) * (b.x * 2.0 - 1.0 ) * depth_bias;
  vec2 newTexCoord = frag_texcoord.xy;// + offset;

  // normal mapping
  vec3 q0  = dFdx(frag_position.xyz);
	vec3 q1  = dFdy(frag_position.xyz);
	vec2 st0 = dFdx(frag_texcoord.st);
	vec2 st1 = dFdy(frag_texcoord.st);

	vec3 S = normalize( q0 * st1.t - q1 * st0.t);
	vec3 T = normalize(-q0 * st1.s + q1 * st0.s);
	mat3 M = mat3(-T, -S, FN);
	vec3 N = normalize(M * (vec3(texture2D(normaltex, newTexCoord.xy)) - vec3(0.5, 0.5, 0.5)));

  // lighting
  vec3 L = normalize(lightpos.xyz - frag_position.xyz);
  vec3 R = normalize(reflect(-L, N));

  vec4 c  = texture2D(colortex, frag_texcoord.xy);

  out_color = c * vec4(dot(N,L) + pow(max(dot(R,N), 0.0), 150.0));
}

