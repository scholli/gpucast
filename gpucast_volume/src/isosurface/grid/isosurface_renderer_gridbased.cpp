/********************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_gridbased.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/grid/isosurface_renderer_gridbased.hpp"

// header, system
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/primitives/plane.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/renderbuffer.hpp>

// header, project
#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>

#include <gpucast/volume/cuda_map_resources.hpp>
#include <gpucast/volume/isosurface/octree/octree.hpp>
#include <gpucast/volume/isosurface/octree/octree_split.hpp>
#include <gpucast/volume/isosurface/octree/split_by_volumecount.hpp>
#include <gpucast/volume/isosurface/octree/split_traversal.hpp>
#include <gpucast/volume/isosurface/octree/info_traversal.hpp>
#include <gpucast/volume/isosurface/octree/serialize_tree_dfs_traversal.hpp>

#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp>
#include <gpucast/math/oriented_boundingbox_random_policy.hpp>
#include <gpucast/math/oriented_boundingbox_axis_aligned_policy.hpp>
#include <gpucast/math/oriented_boundingbox_covariance_policy.hpp>
#include <gpucast/math/oriented_boundingbox_greedy_policy.hpp>

#include <gpucast/core/hyperspace_adapter.hpp>
#include <gpucast/core/util.hpp>
#include <gpucast/core/convex_hull_impl.hpp>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>



namespace gpucast {

extern "C" void invoke_grid_raycasting_kernel  ( renderconfig const&          config,
                                                 bufferinfo const&            info,
                                                 std::array<unsigned,3>       gridsize,
                                                 struct cudaGraphicsResource* input_position_resource,
                                                 struct cudaGraphicsResource* colorbuffer_resource,
                                                 struct cudaGraphicsResource* depthbuffer_resource,
                                                 struct cudaGraphicsResource* cuda_external_texture,
                                                 struct cudaGraphicsResource* cuda_grid_buffer,
                                                 struct cudaGraphicsResource* cuda_face_buffer,
                                                 struct cudaGraphicsResource* cuda_bbox_buffer,
                                                 struct cudaGraphicsResource* cuda_surface_data_buffer,
                                                 struct cudaGraphicsResource* cuda_surface_points_buffer,
                                                 struct cudaGraphicsResource* cuda_volume_data_buffer,
                                                 struct cudaGraphicsResource* cuda_volume_points_buffer,
                                                 struct cudaGraphicsResource* cuda_attribute_data_buffer,
                                                 struct cudaGraphicsResource* cuda_attribute_points_buffer,
                                                 struct cudaGraphicsResource* cuda_matrixbuffer );

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_gridbased::isosurface_renderer_gridbased( int argc, char** argv )
    : isosurface_renderer_structure_based ( argc, argv ),
      _cuda_grid_buffer                   ( 0 ),
      _cuda_face_buffer                   ( 0 ),
      _cuda_bbox_buffer                   ( 0 )
  {}


  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_gridbased::~isosurface_renderer_gridbased()
  {
    unregister_cuda_structure ();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_gridbased::create_data_structure ()
  {
    _create_grid      ();

    _grid->serialize ( _gridbuffer, _facebuffer, _bboxbuffer );
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  isosurface_renderer_gridbased::invoke_ray_casting_kernel ( renderconfig const& config )
  {
    bufferinfo info;

    info.facebuffer_size = _facebuffer.size();
    info.gridbuffer_size = _gridbuffer.size();
    info.bboxbuffer_size = _bboxbuffer.size();

    info.surfacedata_size = _surface_data.size();
    info.surfacepoints_size = _surface_points.size();

    info.volumedata_size  =  _volume_data.size();
    info.volumepoints_size = _volume_points.size();

    info.attributedata_size   = _attribute_data.size();
    info.attributepoints_size = _attribute_points.size();

    invoke_grid_raycasting_kernel ( config,
                                    info,
                                    _gridsize,
                                    _cuda_input_color_depth,
                                    _cuda_output_color,
                                    _cuda_output_depth,
                                    _cuda_external_color_depth,
                                    _cuda_grid_buffer,
                                    _cuda_face_buffer,
                                    _cuda_bbox_buffer,
                                    _cuda_surface_data_buffer,
                                    _cuda_surface_points_buffer,
                                    _cuda_volume_data_buffer,
                                    _cuda_volume_points_buffer,
                                    _cuda_attribute_data_buffer,
                                    _cuda_attribute_points_buffer,
                                    _cuda_matrixbuffer );                                  
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                          
  isosurface_renderer_gridbased::unregister_cuda_structure ()
  {
    unregister_resource ( &_cuda_grid_buffer );
    unregister_resource ( &_cuda_face_buffer );
    unregister_resource ( &_cuda_bbox_buffer );
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_gridbased::write ( std::ostream& os ) const
  {
    isosurface_renderer_structure_based::write(os);

    gpucast::write (os, _gridbuffer);
    gpucast::write (os, _facebuffer);
    gpucast::write (os, _bboxbuffer);

    os.write ( reinterpret_cast<const char*> (&_gridsize[0]), sizeof(_gridsize));
  } 


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_gridbased::read ( std::istream& is )
  {
    isosurface_renderer_structure_based::read(is);

    gpucast::read (is, _gridbuffer);
    gpucast::read (is, _facebuffer);
    gpucast::read (is, _bboxbuffer);

    is.read ( reinterpret_cast<char*> (&_gridsize[0]), sizeof(_gridsize));
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_gridbased::init_structure ()
  {
    _grid_arraybuffer.reset( new gpucast::gl::arraybuffer );
    _bbox_arraybuffer.reset( new gpucast::gl::arraybuffer );
    _face_arraybuffer.reset( new gpucast::gl::arraybuffer );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_gridbased::register_cuda_structure ()
  {
    register_buffer ( &_cuda_grid_buffer, *_grid_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_bbox_buffer, *_bbox_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_face_buffer, *_face_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_gridbased::_create_grid ()
  {
    if ( !_object ) {
      return;
    }

    // create faces from serialized face data
    std::vector<face_ptr> faces;
    _extract_faces(faces);

    // create new grid
    unsigned grid_dimension = std::ceil ( std::pow ( faces.size(), 0.3f ));
    grid_dimension = std::min(grid_dimension, 256U);

    _gridsize[0] = grid_dimension;
    _gridsize[1] = grid_dimension;
    _gridsize[2] = grid_dimension;

    _grid.reset();
    _grid.reset ( new grid(grid_dimension, grid_dimension, grid_dimension) );

    // create octree using face data
    _grid->generate ( _object, faces );

    // create proxy geometry from objects bbox
    _bbox = _object->bbox();
    _corners.clear();
    _bbox.generate_corners ( std::back_inserter(_corners) );
    _raygeneration_geometry->set_vertices  ( _corners[0], _corners[1], _corners[2], _corners[3], _corners[4], _corners[5], _corners[6], _corners[7] );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_gridbased::update_gl_structure ()
  {
    unregister_cuda_structure ();

    _grid_arraybuffer->update     ( _gridbuffer.begin(),  _gridbuffer.end());
    _face_arraybuffer->update     ( _facebuffer.begin(),  _facebuffer.end());
    _bbox_arraybuffer->update     ( _bboxbuffer.begin(),  _bboxbuffer.end());

    std::size_t gridbufsize = _gridbuffer.size() * sizeof(gpucast::gl::vec4u);
    std::size_t facebufsize = _facebuffer.size() * sizeof(gpucast::gl::vec4u);
    std::size_t bboxbufsize = _bboxbuffer.size() * sizeof(gpucast::gl::vec4f);

    std::cout << "Allocating : " << gridbufsize << " Bytes" << std::endl;
    std::cout << "Allocating : " << facebufsize << " Bytes" << std::endl;
    std::cout << "Allocating : " << bboxbufsize << " Bytes" << std::endl;

    std::cout << "total mem  :" << gridbufsize + facebufsize + bboxbufsize << " Bytes" << std::endl;

    register_cuda_structure ();
  }


} // namespace gpucast

