/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_gridbased.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_ISOURFACE_RENDERER_GRID_BASED_HPP
#define GPUCAST_ISOURFACE_RENDERER_GRID_BASED_HPP

// header, system
#include <list>

// header, external

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/isosurface/isosurface_renderer_structure_based.hpp>
#include <gpucast/volume/isosurface/grid/grid.hpp>
#include <gpucast/volume/isosurface/face.hpp>
#include <gpucast/volume/isosurface/renderconfig.hpp>

namespace gpucast 
{

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class GPUCAST_VOLUME isosurface_renderer_gridbased : public isosurface_renderer_structure_based
{
public : // enums, typedefs

public : // c'tor / d'tor

  isosurface_renderer_gridbased    ( int argc, char** argv );
  ~isosurface_renderer_gridbased   ();

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
  
  void                            _create_grid              ();
      
private : // attributes
 
  std::vector<gpucast::gl::vec4u>      _gridbuffer;
  //-----------------------------------------------------------------------------   
  // [min]
  // [max] 
  // [16bit nfaces, 16bit #outer faces]
  // [face_id]
  //-----------------------------------------------------------------------------

  std::vector<gpucast::gl::vec4u>      _facebuffer;
  // [face_id] ...
  //-----------------------------------------------------------------------------
  // [bbox_id]
  // [outer] 
  // [surface_id] 
  // [0 ] 
  //-----------------------------------------------------------------------------

  std::vector<gpucast::gl::vec4f>      _bboxbuffer;  
  // [bbox_id]                             [bbox_id+4]                          [bbox_id+8] [bbox_id+9] [bbox_id+10] 
  //-----------------------------------------------------------------------------    
  // [Orientation00] .. [orientation30]  [OrientationInv00] .. [orientationInv30] [lowx]      [highx]    [centerx]
  // [Orientation01] .. [orientation31]  [OrientationInv01] .. [orientationInv31] [lowy]      [highy]    [centery]
  // [Orientation02] .. [orientation32]  [OrientationInv02] .. [orientationInv32] [lowz]      [highz]    [centerz]
  // [Orientation03] .. [orientation33]  [OrientationInv03] .. [orientationInv33] [0.0]       [0.0]      [0.0]
  //-----------------------------------------------------------------------------

  // grid related data
  std::shared_ptr<grid>                     _grid;
  std::array<unsigned,3>                      _gridsize;
  std::shared_ptr<gpucast::gl::arraybuffer>        _grid_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _face_arraybuffer;
  std::shared_ptr<gpucast::gl::arraybuffer>        _bbox_arraybuffer;

  cudaGraphicsResource*                       _cuda_grid_buffer;
  cudaGraphicsResource*                       _cuda_face_buffer;
  cudaGraphicsResource*                       _cuda_bbox_buffer;
};

} // namespace gpucast

#endif // GPUCAST_ISOURFACE_RENDERER_OCTREE_BASED_HPP

