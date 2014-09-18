/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : target_function.frag
*  project    : gpucast
*  description:
*
********************************************************************************/



///////////////////////////////////////////////////////////////////////////////
float target_function( in vec4 value )
{
  return value.x;
}

///////////////////////////////////////////////////////////////////////////////
float target_function( in float value )
{
  return value;
}

///////////////////////////////////////////////////////////////////////////////
bool is_in_range ( in vec4 value, 
                   in vec4 min_value, 
                   in vec4 max_value )
{
  return  target_function(value) >= target_function(min_value) && 
          target_function(value) <= target_function(max_value);
}

///////////////////////////////////////////////////////////////////////////////
bool is_in_range ( in float value, 
                   in float min_value, 
                   in float max_value )
{
  return  target_function(value) >= target_function(min_value) && 
          target_function(value) <= target_function(max_value);
}

