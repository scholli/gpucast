#ifndef GPUCAST_TRIMMING_BINARY_SEARCH
#define GPUCAST_TRIMMING_BINARY_SEARCH

#include "resources/glsl/common/config.glsl"

// - search_buffer: each element supposed to be in form:  [min max ? ?]
// - value: search value (success if value >= min && value <= max)
// - id: start index in search buffer
// - elements: number of sorted elements to search in
bool
binary_search ( in samplerBuffer search_buffer,
                in float         search_value,
                in int           start_index,
                in int           nelements,
                inout vec4       search_result)
{
  // initialize search
  search_result = vec4(0.0);
  vec4 center_element = vec4(0.0);
  bool found = false;

  int index_min = start_index;
  int index_max = start_index + nelements - 1;

  while (index_min <= index_max)
  {
    // compute center index and fetch center element
    int index_center = index_min + (index_max - index_min) / int(2);

    center_element = texelFetch(search_buffer, index_center);
    gpucast_count_texel_fetch();

    if (search_value >= center_element[0] &&
        search_value <= center_element[1])
    {
      // value found -> report result and return
      search_result = center_element;
      found  = true;
      break;
    } 
    else {
      // value not in range -> reset min/max indices and continue search
      if (search_value < center_element[0])
      {
        index_max = index_center - 1;
      } else {
        index_min = index_center + 1;
      }
    }
  }

  // make sure element was found or not
  if (found)
  {
    return found;
  } else {
    center_element = vec4(0.0);
    return found;
  }
}
   
#endif