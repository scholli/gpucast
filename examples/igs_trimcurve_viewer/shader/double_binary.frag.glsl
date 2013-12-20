#version 330 compatibility
#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable

in vec2     uv_coord; 

uniform int trimid;

uniform samplerBuffer trimdata;
uniform samplerBuffer celldata;
uniform samplerBuffer curvelist;
uniform samplerBuffer curvedata;
uniform sampler1D     transfertexture;

layout (location = 0) out vec4 outcolor; 


///////////////////////////////////////////////////////////////////////////////
vec4 texelFetchUncached ( in samplerBuffer buffer,
                          in int           id,
                          inout vec4       count )
{
  count += vec4 ( 1.0, 0.0, 0.0, 0.0 );
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

    debug.y += float(1);
    tmp = texelFetchUncached(buf, id, debug);

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
   
///////////////////////////////////////////////////////////////////////////////
void
bisect_curve ( in samplerBuffer curvedata_buffer,
               in vec2         uv,
               in vec4         curve,
               inout int       intersections,
               inout int       iterations,
               in float        tolerance,
               in int          max_iterations,
               inout vec4      debug )
{
	float t = 0.0;
  vec4 p  = vec4(0.0);

  int iters = 0;

  int index                    = int(floatBitsToUint(curve[0]));
  int order                    = abs(floatBitsToInt(curve[1]));
  bool horizontally_increasing = floatBitsToInt(curve[1]) > 0;

  debug.y += float(order);

  float tmin = curve[2];
	float tmax = curve[3];

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
trim ( in samplerBuffer partition_buffer,
       in samplerBuffer cell_buffer, 
       in samplerBuffer curvelist_buffer, 
       in samplerBuffer curvedata_buffer, 
       in vec2    uv, 
       in int     id, 
       in int     trim_outer, 
       inout int  iters,
       in float   tolerance,
       in int     max_iterations,
       inout vec4 debug )
{
  debug.y += float(1);
  vec4 domaininfo1 = texelFetchUncached ( partition_buffer, id,   debug );

  int total_intersections  = 0;
  int v_intervals = int ( floatBitsToUint ( domaininfo1[0] ) );

  // if there is no partition in vertical(v) direction -> return
  if ( v_intervals == 0) 
  {
    return false;
  } 
  
  debug.y += float(1);
  vec4 domaininfo2 = texelFetchUncached ( partition_buffer, id+1, debug );

  // classify against whole domain
  if ( uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
       uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2] ) 
  {
    return bool(trim_outer);
  }

  vec4 vinterval = vec4(0.0, 0.0, 0.0, 0.0);
  bool vinterval_found = binary_search ( partition_buffer, uv[1], id + 2, v_intervals, vinterval, debug );

  if ( !vinterval_found ) {
    return bool(trim_outer);
  }

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells      = int(floatBitsToUint(vinterval[3]));

  debug.y += float(1);
  vec4 celllist_info  = texelFetchUncached(cell_buffer, int(celllist_id), debug);
  vec4 cell           = vec4(0.0);

  bool cellfound      = binary_search   (cell_buffer, uv[0], celllist_id + 1, int(ncells), cell, debug );
  if (!cellfound) 
  {
    debug = vec4 ( 1.0, 1.0, 0.0, 0.0 );
    return bool(trim_outer);
  }

  debug.y += float(1);
  vec4 clist                       = texelFetchUncached(curvelist, int(floatBitsToUint(cell[3])), debug);
  total_intersections              = int(floatBitsToUint(cell[2]));
  unsigned int curves_to_intersect = floatBitsToUint(clist[0]);

  debug.y += float(curves_to_intersect);

  for (int i = 1; i <= curves_to_intersect; ++i) 
  {
    vec4 curveinfo = texelFetchUncached ( curvelist_buffer, int(floatBitsToUint(cell[3])) + i, debug );
    bisect_curve ( curvedata_buffer, uv, curveinfo, total_intersections, iters, tolerance, max_iterations, debug );
  }

  if ( mod(total_intersections, 2) == 1 ) 
  {
    return !bool(trim_outer);
  } else {
    return bool(trim_outer);
  }
}



void main(void) 
{ 
  int iterations = 0;
  vec4 debug = vec4 ( 0.0 );
  bool trimmed = trim ( trimdata, celldata, curvelist, curvedata, uv_coord, trimid, 1, iterations, 0.00001, 16, debug );

  if ( trimmed ) 
  {
    //outcolor = texture ( transfertexture, (debug.x / 100.0) );
    outcolor = vec4(1.0, 0.0, 0.0, 1.0 ) * (0.1 + debug.x / 128.0);
  } else {
    //outcolor = texture ( transfertexture, (debug.x / 100.0) );
    outcolor = vec4(0.0, 1.0, 0.0, 1.0 ) * (0.1 + debug.x / 128.0);
  }
}



