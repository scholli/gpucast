/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : serialize_tree_dfs_traversal.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_SERIALIZE_TREE_DFS_TRAVERSAL_HPP
#define GPUCAST_SERIALIZE_TREE_DFS_TRAVERSAL_HPP

// header, system
#include <map>
#include <vector>

#include <boost/noncopyable.hpp>
#include <boost/unordered_map.hpp>

#include <gpucast/math/vec4.hpp>

// header, project
#include <gpucast/volume/isosurface/octree/node.hpp>
#include <gpucast/volume/isosurface/octree/nodevisitor.hpp>


namespace gpucast {

class serialize_tree_dfs_traversal : public nodevisitor
{
public :

  serialize_tree_dfs_traversal();
  ~serialize_tree_dfs_traversal();

  /* virtual */ void              visit                 ( ocnode& ) const;

private : // non-copyable 

  serialize_tree_dfs_traversal ( serialize_tree_dfs_traversal const& );
  serialize_tree_dfs_traversal& operator= ( serialize_tree_dfs_traversal const& );

public : 

  void                              serialize_inner_node  ( ocnode& n ) const;
  void                              serialize_outer_node  ( ocnode& n ) const;
  unsigned                          determine_nodetype    ( ocnode& n ) const;

  unsigned                          serialize_bbox        ( gpucast::math::obbox3f const& ) const;
  unsigned                          serialize_range       ( float min, float max ) const;
  void                              serialize_face        ( face_ptr const& ) const;
  

  void                              finalize              () const;

  std::vector<gpucast::math::vec4u> const&   nodebuffer            () const;
  std::vector<gpucast::math::vec4u> const&   facelistbuffer        () const;
  std::vector<gpucast::math::vec4f> const&   bboxbuffer            () const;
  std::vector<float> const&         limitbuffer           () const;

private : 

  std::vector<std::size_t>*         placeholder;
  std::map<std::size_t, unsigned>*  ocnode_map;  // uniqueid - bufferid
  std::map<face_ptr, gpucast::math::vec4u>*  face_map;     // avoid multiple storage of faces

  // octree structure
  std::vector<gpucast::math::vec4u>*         ocnode_buffer;
  // [ nodetype(parent/no_outer = 0, parent/outer = 1, leaf/no_outer = 2, leaf/outer = 3 ) ] [optional: child0 ] [optional: child5 ]
  // [ attributelimit ocnode id                                                            ] [optional: child1 ] [optional: child6 ]
  // [ first face id                                                                       ] [optional: child2 ] [optional: child7 ]
  // [ number of faces                                                                     ] [optional: child3 ] [optional: child8 ]

  // facelist buffer
  std::vector<gpucast::math::vec4u>*         facelist_buffer;
  //  first face id         first face id + number of faces 
  // [ surface_id ]             [ surface_id ]
  // [ limit id   ]             [ limit id   ]
  // [ obb id     ]             [ obb id     ]
  // [ outer      ]             [ outer      ]

  std::vector<gpucast::math::vec4f>*         bbox_buffer;
  // each oriented boudning box points to this buffer matrix + limits + center
  // [m00] [m10] [m20] [m30] [minv00] [minv10] [minv20] [minv30] [low0] [high0] [cx]
  // [m01] [m11] [m21] [m31] [minv01] [minv11] [minv21] [minv31] [low1] [high1] [cy]
  // [m02] [m12] [m22] [m32] [minv02] [minv12] [minv22] [minv32] [low2] [high2] [cz]
  // [m03] [m13] [m23] [m33] [minv03] [minv13] [minv23] [minv33] [low3] [high3] [ 1]

  // store attribute limits for nodes and faces
  std::vector<float>*               range_buffer;
  // limit id
  //   [min]  [max]  [min]  [max]  [min]  [max] ...

#if 0

  // auxilliary data structure to set pointers from parents to children
  std::shared_ptr<indirection_map_type>    indirection_map;
  std::shared_ptr<volume_index_map>        volume_map;              

  // tree structure
  std::shared_ptr<vec4i_buffer>            treebuffer; 
  // optional data for parent nodes only
  // [ nodetype(parent=0, leaf=1) ] [optional: child0 ] [optional: child5 ]
  // [ limits_id                  ] [optional: child1 ] [optional: child6 ]
  // [ number of elements         ] [optional: child2 ] [optional: child7 ]
  // [ 1st element_id             ] [optional: child3 ] [optional: child8 ]

  std::shared_ptr<vec4i_buffer>            volumelistbuffer;
  // 2 entries per volume
  // [ id_volume ]  [ id_surface]
  // [ orderu ]     [ id_obb    ]
  // [ orderv ]     [ id_limit  ]
  // [ orderw ]     [ ?]

  std::shared_ptr<vec4f_buffer>            boundingboxbuffer;
  // each oriented boudning box points to this buffer matrix + limits + center
  // [m00] [m10] [m20] [m30] [low0] [high0] [cx]
  // [m01] [m11] [m21] [m31] [low1] [high1] [cy]
  // [m02] [m12] [m22] [m32] [low2] [high2] [cz]
  // [m03] [m13] [m23] [m33] [low3] [high3] [ 1]

  std::shared_ptr<vec4f_buffer>            volumebuffer;
  // euclidian coordinates of control points 
  // [p0x] ... [pnx]
  // [p0y] ... [pny]
  // [p0z] ... [pnz]

  std::shared_ptr<vec4f_buffer>           surfacebuffer;
  // euclidian coordinates of boundary surfaces of volumes
  // [p0x] ... [pnx]
  // [p0y] ... [pny]
  // [p0z] ... [pnz]

  // dependent data -> may switch between buffers

  std::shared_ptr<vec4f_buffer_map>        databuffer;
  // data related to each control point
  // [d0x] ... [dnx]
  // [d0y] ... [dny]
  // [d0z] ... [dnz]

  std::shared_ptr<vec4f_buffer_map>        limitbuffer;
  // value (e.g. displacement) limits for a certain boundary element
  // [minx] [maxx] 
  // [miny] [maxy]
  // [minz] [maxz]
#endif
};

} // namespace gpucast

#endif // GPUCAST_SERIALIZE_TREE_DFS_TRAVERSAL_HPP
