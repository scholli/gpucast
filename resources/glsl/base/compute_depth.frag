/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : compute_depth.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

///////////////////////////////////////////////////////////////////////////////
float compute_depth ( in vec4 point_worldcoordinates,
                      in float nearplane,
                      in float farplane ) 
{
  return (point_worldcoordinates.w/point_worldcoordinates.z) * farplane * (nearplane / (farplane - nearplane)) + 0.5 * (farplane + nearplane)/(farplane - nearplane) + 0.5;
}


///////////////////////////////////////////////////////////////////////////////
float compute_depth ( in mat4 modelviewmatrix, 
                      in vec4 point_objectcoordinates, 
                      in float nearplane,
                      in float farplane )
{
  // make sure point has correct homogenous weight
  vec4 p = vec4 (point_objectcoordinates.xyz, 1.0);
  return compute_depth ( modelviewmatrix * p, nearplane, farplane );  
}

