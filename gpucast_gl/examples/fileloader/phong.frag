#version 330 compatibility 
 
in vec4 fragnormal;    
in vec4 fragposition;  
in vec4 fragtexcoord;  

uniform vec4 lightpos;

uniform vec4 ka;
uniform vec4 kd;
uniform vec4 ks;

uniform float opacity;
uniform float shininess;

layout (location = 0) out vec4 color;
 
void main(void)  
{  
  vec3 L = normalize(lightpos.xyz - fragposition.xyz);   
  vec3 V = normalize(-fragposition.xyz);   
  vec3 N = normalize(fragnormal.xyz);  

  // switch normal to viewer if necessary
  if (dot(N,V) < 0.0) {  
    N *= -1.0;
  }

  vec4 lightcolor = vec4(1.0, 1.0, 1.0, 1.0);

  // amibent
  vec4 ambient = ka * lightcolor;

  // diffuse illumination
  vec4 diffuse = kd * lightcolor * vec4(dot(V, N));

  // specular reflection
  vec3 R = reflect(-L, N);
  vec4 specular = ks * lightcolor * max(0.0, pow(dot(R,V), shininess));

  // light attenuation
  float attenuation = min(1.0, 100.0 / length(fragposition.xyz));  

  // add up colors
  color = ambient + attenuation * (diffuse + specular);
  color.w = opacity;
}

