#version 330 compatibility
#extension GL_EXT_gpu_shader4 : enable
    
in vec4 fragposition;  
in vec4 fragtexcoord;  

uniform vec4 lightpos;
uniform int mode;

uniform sampler2D ambient;
uniform sampler2D diffuse;
uniform sampler2D specular;
uniform sampler2D normal;

layout (location = 0) out vec4 color;
 
void main(void)  
{  
  vec4 ka       = texture2D(ambient, fragtexcoord.xy);
  vec4 kd       = texture2D(diffuse, fragtexcoord.xy);
  vec4 ks       = texture2D(specular, fragtexcoord.xy);
  vec4 n        = texture2D(normal, fragtexcoord.xy);
  
  vec4 p        = vec4(ka.w, kd.w, n.w, 1.0);
  
  
  if (mode == 0)
  {
    vec3 V = normalize(-p.xyz);
    vec3 N = normalize(n.xyz);   
    vec3 L = normalize(lightpos.xyz - p.xyz);
    vec3 R = reflect(-L, N);
    
    vec4 amb  = ka;
    vec4 dif  = kd * dot(L,N);
    vec4 spec = ks * max(0.0, dot(R,V));
    
    color = amb + (dif + spec);
  }
  
  if (mode == 1) 
  {
    color = ka;
  }
  
  if (mode == 2) 
  {
    color = kd;
  }
  
  if (mode == 3) 
  {
    color = ks;
  }
  
  if (mode == 4) 
  {
    color = n;
  }
}


