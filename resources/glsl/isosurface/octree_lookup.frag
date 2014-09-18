/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : octree_lookup.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

ivec4 octree_lookup ( in isamplerBuffer tree,
                      in vec3           coords,
                      in vec3           tree_min,
                      in vec3           tree_max,
                      out int           depth )
{
  vec3  local_coords  = coords - tree_min;
  int   index         = 0;
  int   color_index   = 0;
  ivec4 node          = ivec4(0);

  depth   = 0;

  vec3 tree_size   = tree_max - tree_min;
  vec3 tree_center = tree_size / 2.0;

  while ( node.x == 0 )
  {
    node          = texelFetchBuffer(tree, index);

    if ( node.x == 0 )
    {
      // compute index of child node index
      index         += 1 + int(local_coords[2] > tree_center[2]);
      color_index    = int(local_coords[0] > tree_center[0]) + 2 * int(local_coords[1] > tree_center[1]);

      // fetch child node index
      ivec4 children = texelFetchBuffer(tree, index);
      index          = children[color_index]; 

      local_coords  *= 2.0;      

      if (local_coords[0] > tree_size[0]) local_coords[0] -= tree_size[0];
      if (local_coords[1] > tree_size[1]) local_coords[1] -= tree_size[1];
      if (local_coords[2] > tree_size[2]) local_coords[2] -= tree_size[2];

      ++depth;
    }
  }

  return node;
}

