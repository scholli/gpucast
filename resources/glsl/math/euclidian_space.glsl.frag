/*********************************************************************
 * project point into hyperspace p[x,y,z,w] = [x/w, y/w, z/w, w]
 *********************************************************************/
vec4 euclidian_space(in vec4 point)
{
  // backup weight 
  float weight = point[3];
  
  // reproject to w=1-plane
  point /= weight;
  
  // reset weight 
  point[3] = weight;
  
  return point;
}


vec3 euclidian_space(in vec3 point)
{
  // backup weight 
  float weight = point[2];
  
  // reproject to w=1-plane
  point /= weight;
  
  // reset weight 
  point[2] = weight;
  
  return point;
}

