#version 420 core
#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable
     
in vec2     uv_coord; 
     
uniform int trimid;

uniform samplerBuffer sampler_partition;
uniform samplerBuffer sampler_contourlist;
uniform samplerBuffer sampler_curvelist;
uniform samplerBuffer sampler_curvedata;
uniform samplerBuffer sampler_pointdata;

uniform sampler1D     transfertexture;

layout (location = 0) out vec4 outcolor; 

///////////////////////////////////////////////////////////////////////////////
uvec4 intToUInt4 ( uint input )
{
  uvec4 result;
  result.w = (input & 0xFF000000) >> 24U;
  result.z = (input & 0x00FF0000) >> 16U;
  result.y = (input & 0x0000FF00) >> 8U;
  result.x = (input & 0x000000FF);
  return result;
}

///////////////////////////////////////////////////////////////////////////////
void intToUint8_24 ( in  uint input,
                     out uint a,
                     out uint b )
{
  b = (input & 0xFFFFFF00) >> 8U;
  a = (input & 0x000000FF);
}

///////////////////////////////////////////////////////////////////////////////
vec4 texelFetchUncached ( in samplerBuffer buffer,
                          in int           id,
                          inout vec4       count )
{
  count.x += 1;
  return texelFetch ( buffer, id );
}
     
///////////////////////////////////////////////////////////////////////////////
void 
evaluateCurve ( in samplerBuffer data, 
                in int index, 
                in int order, 
                in float t, 
                out vec4 p,
                inout vec4 debug ) 
{
  int deg = order - 1;
  float u = 1.0 - t;
  
  float bc = 1.0;
  float tn = 1.0;
  p  = texelFetchUncached(data, index, debug);
  p *= u;

  if (order > 2) {
    for (int i = 1; i <= deg - 1; ++i) {
      tn *= t;
      bc *= (float(deg-i+1) / float(i));
      p = (p + tn * bc * texelFetchUncached(data, index + i, debug)) * u;
    } 
    p += tn * t * texelFetchUncached(data, index + deg, debug);
  } else {
    /* linear piece*/
    p = mix(texelFetchUncached(data, index, debug), texelFetchUncached(data, index + 1, debug), t);
  }
    
  /* project into euclidian coordinates */
  p[0] = p[0]/p[2];
  p[1] = p[1]/p[2];
}

///////////////////////////////////////////////////////////////////////////////
bool
binary_search ( in samplerBuffer buf,
                in float         value,
                in int           id,
                in int           intervals,
                inout vec4       result,
                inout vec4       debug )
{
  result = vec4(0.0);

  int id_min = id;
  int id_max = id + intervals - 1;

  vec4 tmp = vec4(0.0);
  bool found = false;

  while ( id_min <= id_max )
  {
    int id = id_min + (id_max - id_min) / int(2);

    tmp = texelFetchUncached(buf, id, debug);

    if (value > tmp[0] && value < tmp[1])
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

///////////////////////////////////////////////////////////////////////////////
bool
contour_binary_search ( in samplerBuffer buffer,
                        in vec2          uv,
                        in int           id,
                        in int           intervals,
                        in bool          uincreasing,
                        inout int        intersections,
                        inout int        curveindex,
                        inout vec4       debug )
{
  int id_min = id;
  int id_max = id + intervals - 1;

  vec4 tmp = vec4(0.0);
  bool found = false;

  while ( id_min <= id_max )
  {
    int id = id_min + (id_max - id_min) / int(2);
    tmp = texelFetchUncached(buffer, id, debug);

    if ( uv[1] >= tmp[0] && uv[1] <= tmp[1])
    {
      if ( uv[0] >= tmp[2] && uv[0] <= tmp[3] )
      {
        curveindex = id;
        found = true;
      } else { 
        if ( uv[0] < tmp[2] ) 
        {
          ++intersections;
        }
      }
      break;
    } else {
      
      if ( ( uv[1] < tmp[1] && uv[0] > tmp[3] &&  uincreasing ) || 
           ( uv[1] < tmp[0] && uv[0] > tmp[2] &&  uincreasing ) || 
           ( uv[1] > tmp[0] && uv[0] > tmp[3] && !uincreasing ) || 
           ( uv[1] > tmp[1] && uv[0] > tmp[2] && !uincreasing ) )
      {
        break;
      }
      
      if ( ( uv[1] > tmp[0] && uv[0] < tmp[2] &&  uincreasing ) || 
           ( uv[1] > tmp[1] && uv[0] < tmp[3] &&  uincreasing ) ||
           ( uv[1] < tmp[1] && uv[0] < tmp[2] && !uincreasing ) ||
           ( uv[1] < tmp[0] && uv[0] < tmp[3] && !uincreasing ))
      {
        ++intersections;
        break;
      }

      if ( uv[1] < tmp[0] ) 
      {
        id_max = id - 1;
      } else {
        id_min = id + 1;
      }
    }
  }

  return found;
}

///////////////////////////////////////////////////////////////////////////////
bool
bisect_contour ( in samplerBuffer buffer,
                 in vec2          uv,
                 in int           id,
                 in int           intervals,
                 in bool          uincreasing,
                 inout int        intersections,
                 out uint         index,
                 out uint         order,
                 inout vec4       debug )
{
  int id_min = id;
  int id_max = id + intervals - 1;

  index = 0;
  order = 0;

  bool preclassified = false;

  //for ( int k = 0; k != intervals; ++k )
  while ( id_min <= id_max && !preclassified )
  {
    int id    = id_min + (id_max - id_min) / int(2);
    vec4 p    = texelFetchUncached(buffer, id, debug);

    vec2  u   = unpackHalf2x16 ( floatBitsToUint ( p.z ) );
    uvec4 tmp = intToUInt4 ( floatBitsToUint ( p.w ) );

    // point inside 
    if ( uv[1] >= p[0] && 
         uv[1] <= p[1] && 
         uv[0] >= u[0] &&
         uv[0] <= u[1] )
    {
      intToUint8_24 ( floatBitsToUint ( p.w ), order, index );
      preclassified = true;
      break;
    }
    
    // pre-classification of non-intersection
	  if ( (!uincreasing && uv[0] > u[0] && uv[1] > p[1]) || 
         (!uincreasing && uv[0] > u[1] && uv[1] > p[0]) ||
         ( uincreasing && uv[0] > u[0] && uv[1] < p[0]) ||
         ( uincreasing && uv[0] > u[1] && uv[1] < p[1]) )
    {
      break;
	  }
    
    // pre-classification of intersection
	  if ( (!uincreasing && uv[0] < u[0] && uv[1] < p[1]) ||
         (!uincreasing && uv[0] < u[1] && uv[1] < p[0]) ||
         ( uincreasing && uv[0] < u[0] && uv[1] > p[0]) ||
         ( uincreasing && uv[0] < u[1] && uv[1] > p[1]) )
    {
      ++intersections;
      break;
	  }
    
    // next step in binary search
    if (uv[1] < p[1]) 
    {
	    id_max = id - 1;
	  } else {
	    id_min = id + 1;
	  }
  }

  return preclassified;
}
   
///////////////////////////////////////////////////////////////////////////////
void
bisect_curve ( in samplerBuffer curvedata_buffer,
               in vec2          uv,
               in int           index,
               in int           order,
               in bool          horizontally_increasing,
               in float         tmin,
               in float         tmax,
               inout int        intersections,
               inout int        iterations,
               in float         tolerance,
               in int           max_iterations,
               inout vec4       debug )
{
	float t = 0.0;
  vec4 p  = vec4(0.0);

  int iters = 0;

  for (int i = 0; i < max_iterations; ++i)
  {
    ++iterations;
	  t = (tmax + tmin) / 2.0;
	  evaluateCurve ( curvedata_buffer, index, order, t, p, debug );

    if ( length ( uv - p.xy ) < tolerance )
    {
      break;
    }

	  if (uv[1] > p[1]) 
    {
	    tmin = t;
	  } else {
	    tmax = t;
	  }

	  if ( (!horizontally_increasing && uv[0] > p[0] && uv[1] > p[1] )||
         ( horizontally_increasing && uv[0] > p[0] && uv[1] < p[1] ) ) 
    {
      break;
	  }

	  if ( (!horizontally_increasing && uv[0] < p[0] && uv[1] < p[1]) ||
         (horizontally_increasing && uv[0] < p[0] && uv[1] > p[1]) )
    {
      ++intersections;
      break;
	  }
	}
}

bool
trim ( in samplerBuffer partition,
       in samplerBuffer contourlist,
       in samplerBuffer curvelist,
       in samplerBuffer curvedata,
       in samplerBuffer pointdata,
       in vec2          uv, 
       in int           id, 
       in int           trim_outer, 
       inout int        iters,
       in float         tolerance,
       in int           max_iterations,
       inout vec4       debug )
{
  int total_intersections  = 0;
  int v_intervals          = int ( floatBitsToUint ( texelFetchUncached ( partition, id, debug ).x ) );

  // if there is no partition in vertical(v) direction -> return
  if ( v_intervals == 0) 
  {
    return false;
  }
    
  vec4 domaininfo2 = texelFetchUncached ( partition, id+1, debug );

  // classify against whole domain
  if ( uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
       uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2] ) 
  {
    return bool(trim_outer);
  }

  vec4 vinterval = vec4(0.0);
  bool vinterval_found = binary_search ( partition, uv[1], id + 2, v_intervals, vinterval, debug );

  //if ( !vinterval_found ) {
  //  return bool(trim_outer);
  //}

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells      = int(floatBitsToUint(vinterval[3]));

  vec4 celllist_info  = texelFetchUncached(partition, int(celllist_id), debug);
  vec4 cell           = vec4(0.0);

  bool cellfound      = binary_search (partition, uv[0], celllist_id + 1, int(ncells), cell, debug );
  //if (!cellfound) 
  //{
  //  return bool(trim_outer);
  //}

  uvec4 type_ncontours     = intToUInt4 ( floatBitsToUint(cell[2]) );
  total_intersections      = int(type_ncontours.y);
  int overlapping_contours = int(type_ncontours.z);

  int contourlist_id       = int(floatBitsToUint(cell[3]));

  for ( int i = 0; i < overlapping_contours; ++i )
  {
    vec2 contour    = texelFetchUncached ( contourlist, contourlist_id + i, debug ).xy;

    uvec4 ncurves_uincreasing = intToUInt4 ( floatBitsToUint(contour.x) );
    bool contour_uincreasing  = ncurves_uincreasing.y > 0;
    int curves_in_contour     = int(ncurves_uincreasing.x);
    int  curvelist_id         = int(floatBitsToUint(contour.y));

//#define NO_EXTRA_CURVEINFO_BUFFER
#ifdef NO_EXTRA_CURVEINFO_BUFFER
    uint  curveid      = 0;
    uint  curveorder   = 0;
    bool process_curve = bisect_contour ( curvelist, 
                                          uv, 
                                          curvelist_id, 
                                          curves_in_contour, 
                                          contour_uincreasing, 
                                          total_intersections,
                                          curveid,
                                          curveorder,
                                          debug );
    if ( process_curve ) 
    {
      int iters = 0;
      bisect_curve ( pointdata, uv, int(curveid), int(curveorder), contour_uincreasing, 0.0f, 1.0f, total_intersections, iters, tolerance, max_iterations, debug );
    }
#else 
    int curveid       = 0;
    bool process_curve = contour_binary_search ( curvelist, 
                                                 uv, 
                                                 curvelist_id, 
                                                 curves_in_contour, 
                                                 contour_uincreasing, 
                                                 total_intersections,
                                                 curveid,
                                                 debug );
    if ( process_curve ) 
    {
      int iters = 0;
      float curveinfo = texelFetch (curvedata, curveid).x;
      uint pointid    = 0;
      uint curveorder = 0;
      intToUint8_24 ( floatBitsToUint ( curveinfo ), curveorder, pointid );
      bisect_curve ( pointdata, uv, int(pointid), int(curveorder), contour_uincreasing, 0.0f, 1.0f, total_intersections, iters, tolerance, max_iterations, debug );
    }
#endif
  }
  
  return ( (mod(total_intersections, 2) == 1) != bool(trim_outer) );
}


void main(void) 
{ 
  int iterations = 0;
  vec4 debug = vec4 ( 0.0 );
  bool trimmed = trim ( sampler_partition, 
                        sampler_contourlist,
                        sampler_curvelist,
                        sampler_curvedata,
                        sampler_pointdata,
                        uv_coord, 
                        trimid, 
                        1, 
                        iterations, 
                        0.00001, 
                        16, 
                        debug );

  if ( trimmed ) 
  {
    //outcolor = debug;
    outcolor = vec4(1.0, 0.0, 0.0, 1.0 ) * (0.1 + debug.x / 128.0);
    //outcolor = texture ( transfertexture, (debug.x / 100.0) );
  } else {
    //outcolor = debug;
    outcolor = vec4(0.0, 1.0, 0.0, 1.0 ) * (0.1 + debug.x / 128.0);
    //outcolor = texture ( transfertexture, (debug.x / 100.0) );
  }
}



