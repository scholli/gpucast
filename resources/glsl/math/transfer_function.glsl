#ifndef GPUCAST_TRANSFER_FUNCTION
#define GPUCAST_TRANSFER_FUNCTION

vec4 transfer_function(in float relative_input)
{
  vec4 low = vec4(0.0, 0.0, 1.0, 1.0);
  vec4 normal = vec4(0.0, 1.0, 0.0, 1.0);
  vec4 high = vec4(1.0, 1.0, 0.0, 1.0);
  vec4 critical = vec4(1.0, 0.0, 0.0, 1.0);

  if (relative_input < 0.33) {
    return mix(low, normal, relative_input / 0.33);
  }
  else if (relative_input >= 0.33 && relative_input < 0.66) {
    return mix(normal, high, (relative_input - 0.33) / 0.33);
  }
  else if (relative_input >= 0.66 && relative_input <= 1.0) {
    return mix(high, critical, (relative_input - 0.66) / 0.33);
  }
  return vec4(0.0);
}

#endif