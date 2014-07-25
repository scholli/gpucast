#ifndef LIBGPUCAST_TARGET_FUNCTION_H
#define LIBGPUCAST_TARGET_FUNCTION_H

///////////////////////////////////////////////////////////////////////////////
__device__ inline
float target_function ( float value )
{
  return value;
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
bool is_in_range ( float value, 
                   float min_value, 
                   float max_value )
{
  return  target_function(value) >= target_function(min_value) && 
          target_function(value) <= target_function(max_value);
}

#endif // LIBGPUCAST_TARGET_FUNCTION_H