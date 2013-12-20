#version 120 
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D left;
uniform sampler2D right;

uniform int width;
uniform int height;

void main(void)
{
  vec2 texCoord = gl_TexCoord[0].xy;
  vec2 offset = vec2(1.0/float(width), 1.0/float(height));
  
  /* apply blur filter*/
  vec4 pixel_left_eye = texture2D(left, vec2(texCoord.x, texCoord.y));        

  } else { 
    gl_FragColor = pixel_rght_eye;
  }
}



