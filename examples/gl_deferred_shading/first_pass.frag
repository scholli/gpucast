#version 330 compatibility 
#extension GL_EXT_gpu_shader4 : enable

in vec4 fragnormal;    
in vec4 fragposition;  
in vec4 fragtexcoord;  

uniform vec4 ka;
uniform vec4 kd;
uniform vec4 ks;

uniform float opacity;
uniform float shininess;

layout (location = 0) out vec4  ambient;
layout (location = 1) out vec4  diffuse;
layout (location = 2) out vec4  specular;
layout (location = 3) out vec4  normal;

void main(void)  
{  
  vec3 V = normalize(-fragposition.xyz);   
  vec3 N = normalize( fragnormal.xyz);  

  // switch normal to viewer if necessary
  if (dot(N,V) < 0.0) {  
    N *= -1.0;
  }
  
  // write all necessary information for deferred shading
  ambient = vec4(ka.xyz, fragposition.x);
  diffuse = vec4(kd.xyz , fragposition.y);
  specular = vec4(ks.xyz, shininess);  
  normal = vec4(N, fragposition.z);
}

