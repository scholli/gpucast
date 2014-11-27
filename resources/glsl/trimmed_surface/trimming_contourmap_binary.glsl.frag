///////////////////////////////////////////////////////////////////////////////
uvec4 intToUInt4 ( in uint input )
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
bool
bisect_contour ( in samplerBuffer data,
                 in vec2          uv,
                 in int           id,
                 in int           intervals,
                 in bool          uincreasing,
                 inout int        intersections,
                 out uint         index,
                 out uint         order )
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
    vec4 p    = texelFetch(data, id);
    
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
bool
contour_binary_search ( in samplerBuffer data,
                        in vec2          uv,
                        in int           id,
                        in int           intervals,
                        in bool          uincreasing,
                        inout int     intersections,
                        inout int     curveindex )
{
  int id_min = id;
  int id_max = id + intervals - 1;

  vec4 tmp = vec4(0.0);
  bool found = false;

  while ( id_min <= id_max )
  {
    int id = id_min + (id_max - id_min) / int(2);
    tmp = texelFetch(data, id );

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

#if 0
      if ( uv[1] < tmp[0] && uv[0] > tmp[3] && uincreasing || 
           uv[1] > tmp[1] && uv[0] > tmp[3] && !uincreasing )
      {
        break;
      }
      
      if ( uv[1] > tmp[1] && uv[0] < tmp[2] && uincreasing || 
           uv[1] < tmp[0] && uv[0] < tmp[2] && !uincreasing )
      {
        ++intersections;
        break;
      }
#else
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
#endif

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


bool
trim ( in samplerBuffer trimpartition,
       in samplerBuffer contourlist,
       in samplerBuffer curvelist,
       in samplerBuffer curvedata,
       in samplerBuffer pointdata,
       in vec2          uv, 
       in int           id, 
       in int           trim_outer, 
       inout int     iters,
       in float         tolerance,
       in int           max_iterations )
{
  int total_intersections  = 0;
  int v_intervals          = int ( floatBitsToUint ( texelFetch ( trimpartition, id ).x ) );

  // if there is no partition in vertical(v) direction -> return
  if ( v_intervals == 0) 
  {
    return false;
  }
    
  vec4 domaininfo2 = texelFetch ( trimpartition, id+1 );

  // classify against whole domain
  if ( uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
       uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2] ) 
  {
    return bool(trim_outer);
  }

  vec4 vinterval = vec4(0.0, 0.0, 0.0, 0.0);
  bool vinterval_found = binary_search ( trimpartition, uv[1], id + 2, v_intervals, vinterval );

  //if ( !vinterval_found ) {
  //  return bool(trim_outer);
  //}

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells      = int(floatBitsToUint(vinterval[3]));

  vec4 celllist_info  = texelFetch(trimpartition, int(celllist_id) );
  vec4 cell           = vec4(0.0);
  bool cellfound      = binary_search   (trimpartition, uv[0], celllist_id + 1, int(ncells), cell );

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
    vec2 contour    = texelFetch ( contourlist, contourlist_id + i ).xy;

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
                                          curveorder );
    if ( process_curve ) 
    {
      int iters = 0;
      bisect_curve ( pointdata, uv, int(curveid), int(curveorder), contour_uincreasing, 0.0f, 1.0f, total_intersections, iters, tolerance, max_iterations );
    }
#else 
    int curveid       = 0;
    bool process_curve = contour_binary_search ( curvelist, 
                                                 uv, 
                                                 curvelist_id, 
                                                 curves_in_contour, 
                                                 contour_uincreasing, 
                                                 total_intersections,
                                                 curveid );
    if ( process_curve ) 
    {
      int iters = 0;
      float curveinfo = texelFetch (curvedata, curveid).x;
      uint pointid    = 0;
      uint curveorder = 0;
      intToUint8_24 ( floatBitsToUint ( curveinfo ), curveorder, pointid );
      bisect_curve ( pointdata, uv, int(pointid), int(curveorder), contour_uincreasing, 0.0f, 1.0f, total_intersections, iters, tolerance, max_iterations );
    }
#endif
  }
  
  return ( (mod(total_intersections, 2) == 1) != bool(trim_outer) );
  
}



