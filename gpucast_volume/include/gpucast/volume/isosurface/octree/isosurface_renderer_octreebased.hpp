/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_octreebased.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_ISOURFACE_RENDERER_OCTREE_BASED_HPP
#define GPUCAST_ISOURFACE_RENDERER_OCTREE_BASED_HPP

// header, system
#include <list>

// header, external
#include <memory>
#include <boost/unordered_map.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/isosurface/isosurface_renderer_structure_based.hpp>
#include <gpucast/volume/isosurface/renderconfig.hpp>
#include <gpucast/volume/isosurface/octree/node.hpp>
#include <gpucast/volume/isosurface/octree/octree.hpp>
#include <gpucast/volume/isosurface/octree/serialize_tree_dfs_traversal.hpp>

namespace gpucast 
{

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class GPUCAST_VOLUME isosurface_renderer_octreebased : public isosurface_renderer_structure_based
{
public : // enums, typedefs

public : // c'tor / d'tor

  isosurface_renderer_octreebased    ( int argc, char** argv );
  ~isosurface_renderer_octreebased   ();

public : // methods

  virtual void                    register_cuda_structure   ();
  virtual void                    unregister_cuda_structure ();

  virtual void                    invoke_ray_casting_kernel ( renderconfig const& config );
  virtual void                    create_data_structure     ();

  virtual void                    update_gl_structure       ();

  virtual void                    init_structure            ();

  virtual void                    write                     ( std::ostream& os ) const;
  virtual void                    read                      ( std::istream& is );

private : // auxilliary methods

  void                            _create_octree            ();

private : // attributes

  std::vector<gpucast::math::vec4u>      _nodebuffer;
  //-----------------------------------------------------------------------------   
  // [nodetype]       [childnodeid_uvw] [childnodeid_Uvw]
  // [limitbuffer_id] [childnodeid_uvW] [childnodeid_UvW]
  // [0]              [childnodeid_uVw] [childnodeid_UVw]
  // [0]              [childnodeid_uVW] [childnodeid_UVW]

  std::vector<gpucast::math::vec4u>      _facebuffer;
  //-----------------------------------------------------------------------------   
  // [surface_data_id] 
  // [limitbuffer_id]
  // [bbox_id]   
  // [is_outer] 
  //-----------------------------------------------------------------------------

  std::vector<gpucast::math::vec4f>      _bboxbuffer;
  // [bbox_id]                             [bbox_id+4]                          [bbox_id+8] [bbox_id+9] [bbox_id+10] 
  //-----------------------------------------------------------------------------    
  // [Orientation00] .. [orientation30]  [OrientationInv00] .. [orientationInv30] [lowx]      [highx]    [centerx]
  // [Orientation01] .. [orientation31]  [OrientationInv01] .. [orientationInv31] [lowy]      [highy]    [centery]
  // [Orientation02] .. [orientation32]  [OrientationInv02] .. [orientationInv32] [lowz]      [highz]    [centerz]
  // [Orientation03] .. [orientation33]  [OrientationInv03] .. [orientationInv33] [0.0]       [0.0]      [0.0]
  //-----------------------------------------------------------------------------

  std::vector<float>            _limitbuffer; 
  //-----------------------------------------------------------------------------   
  // [limitbuffer_id] [limitbuffer_id+1]
  //-----------------------------------------------------------------------------   
  //   [attrib_min]     [attrib_max]     ... 
  //-----------------------------------------------------------------------------

  // octree related data
  std::shared_ptr<octree>                   _octree;
  std::shared_ptr<gpucast::gl::arraybuffer>        _octree_node_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _octree_face_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _octree_bbox_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _octree_limit_arraybuffer;

  cudaGraphicsResource*                       _cuda_octree_node_buffer;
  cudaGraphicsResource*                       _cuda_octree_face_buffer;
  cudaGraphicsResource*                       _cuda_octree_bbox_buffer;
  cudaGraphicsResource*                       _cuda_octree_limit_buffer;
};

} // namespace gpucast

#endif // GPUCAST_ISOURFACE_RENDERER_OCTREE_BASED_HPP

