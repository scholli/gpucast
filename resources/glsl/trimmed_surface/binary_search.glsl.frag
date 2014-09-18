/*****************************************************************************
 * binary search a LIST of sorted intervals :
 *
 *  buf       - samplerbuffer with sorted intervals 
 *              buf[i][0] is minimum of i-th interval
 *              buf[i][1] is maximum of i-th interval
 *
 *  value     - value to look for (in which interval it is)
 *
 *  id        - index of first interval
 *
 *  intervals - number of intervals
 *
 *  result    - texture entry buf[i] in which the result is in
 *
 ****************************************************************************/

bool
binary_search ( in samplerBuffer buf,
                in float         value,
                in int           id,
                in int           intervals,
                inout vec4       result )
{
  result = vec4(0.0);

  int id_min = id;
  int id_max = id + intervals - 1;

  vec4 tmp = vec4(0.0);
  bool found = false;

  while ( id_min <= id_max )
  {
    int id = id_min + (id_max - id_min) / int(2);

    tmp = texelFetch(buf, id);

    if (value >= tmp[0] && value <= tmp[1])
    {
      result = tmp;
      found    = true;
      break;
    } else {
      if ( value < tmp[0] ) 
      {
        id_max = id - 1;
      } else {
        id_min = id + 1;
      }
    }
  }

  if (found)
  {
    return found;
  } else {
    result = vec4(0.0);
    return found;
  }
}

