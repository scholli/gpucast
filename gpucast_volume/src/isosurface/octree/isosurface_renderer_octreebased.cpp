/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_octreebased.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/isosurface_renderer_octreebased.hpp"

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

extern "C" void invoke_octree_raycasting_kernel  ( renderconfig const&          config,
                                                   gpucast::bufferinfo const&   info,
                                                   struct cudaGraphicsResource* input_position_resource,
                                                   struct cudaGraphicsResource* colorbuffer_resource,
                                                   struct cudaGraphicsResource* depthbuffer_resource,
                                                   struct cudaGraphicsResource* cuda_external_texture,
                                                   struct cudaGraphicsResource* cuda_octree_node_buffer,
                                                   struct cudaGraphicsResource* cuda_octree_face_buffer,
                                                   struct cudaGraphicsResource* cuda_octree_bbox_buffer,
                                                   struct cudaGraphicsResource* cuda_octree_limit_buffer,
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
  isosurface_renderer_octreebased::isosurface_renderer_octreebased( int argc, char** argv )
    : isosurface_renderer_structure_based ( argc, argv ),
      _cuda_octree_node_buffer            ( 0 ),
      _cuda_octree_face_buffer            ( 0 ),
      _cuda_octree_bbox_buffer            ( 0 ),
      _cuda_octree_limit_buffer           ( 0 )
  {}


  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_octreebased::~isosurface_renderer_octreebased()
  {
    unregister_cuda_structure ();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_octreebased::create_data_structure ()
  {
    _create_octree      ();

    serialize_tree_dfs_traversal serializer;
    _octree->accept(serializer);

    _nodebuffer  = serializer.nodebuffer();
    _facebuffer  = serializer.facelistbuffer();
    _bboxbuffer  = serializer.bboxbuffer();
    _limitbuffer = serializer.limitbuffer();
  }




  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  isosurface_renderer_octreebased::invoke_ray_casting_kernel ( renderconfig const& config )
  {
    bufferinfo info;

    info.facebuffer_size      = _facebuffer.size();
    info.octree_size          = _nodebuffer.size();
    info.bboxbuffer_size      = _bboxbuffer.size();

    info.surfacedata_size     = _surface_data.size();
    info.surfacepoints_size   = _surface_points.size();

    info.volumedata_size      =  _volume_data.size();
    info.volumepoints_size    = _volume_points.size();

    info.attributedata_size   = _attribute_data.size();
    info.attributepoints_size = _attribute_points.size();

    invoke_octree_raycasting_kernel ( config,
                                      info,
                                      _cuda_input_color_depth,
                                      _cuda_output_color,
                                      _cuda_output_depth,
                                      _cuda_external_color_depth,
                                      _cuda_octree_node_buffer,
                                      _cuda_octree_face_buffer,
                                      _cuda_octree_bbox_buffer,
                                      _cuda_octree_limit_buffer,
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
  isosurface_renderer_octreebased::unregister_cuda_structure ()
  {
    unregister_resource ( &_cuda_octree_node_buffer );
    unregister_resource ( &_cuda_octree_face_buffer );
    unregister_resource ( &_cuda_octree_bbox_buffer );
    unregister_resource ( &_cuda_octree_limit_buffer );
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_octreebased::write ( std::ostream& os ) const
  {
    isosurface_renderer_structure_based::write(os);

    gpucast::write (os, _nodebuffer);
    gpucast::write (os, _facebuffer);
    gpucast::write (os, _bboxbuffer);
    gpucast::write (os, _limitbuffer);        
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_octreebased::read ( std::istream& is )
  {
    isosurface_renderer_structure_based::read(is);

    gpucast::read (is, _nodebuffer);
    gpucast::read (is, _facebuffer);
    gpucast::read (is, _bboxbuffer);
    gpucast::read (is, _limitbuffer);
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_octreebased::init_structure ()
  {
    _octree_node_arraybuffer.reset  ( new gpucast::gl::arraybuffer );
    _octree_bbox_arraybuffer.reset  ( new gpucast::gl::arraybuffer );
    _octree_face_arraybuffer.reset  ( new gpucast::gl::arraybuffer );
    _octree_limit_arraybuffer.reset ( new gpucast::gl::arraybuffer );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_octreebased::register_cuda_structure ()
  {
    register_buffer ( &_cuda_octree_node_buffer,  *_octree_node_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_octree_bbox_buffer,  *_octree_bbox_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_octree_face_buffer,  *_octree_face_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_octree_limit_buffer, *_octree_limit_arraybuffer, cudaGraphicsRegisterFlagsReadOnly );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_octreebased::_create_octree ()
  {
    if ( !_object ) {
      return;
    }

    std::string path            = boost::filesystem::path(_object->parent()->name()).parent_path().string();
    std::string basename        = boost::filesystem::basename(_object->parent()->name());
    std::string octree_filename = path + "/" + basename + "_" +
                                  boost::lexical_cast<std::string>(_max_octree_depth) + "_" +
                                  boost::lexical_cast<std::string>(_max_volumes_per_node) + ".ocb";

    // create new octree
    _octree.reset();
    _octree.reset ( new octree );

    // create faces from serialized face data
    std::vector<face_ptr> faces;
    _extract_faces(faces);

    // create octree using face data
    split_criteria_ptr split_criteria ( new split_by_volumecount ( _max_octree_depth, _max_volumes_per_node ) );
    split_traversal splitter ( split_criteria );
    _octree->generate ( _object, faces, splitter, 0.001f );

    // create proxy geometry from octree's bbox
    _bbox = _octree->root()->boundingbox();
    _corners.clear();
    _bbox.generate_corners ( std::back_inserter(_corners) );
    _raygeneration_geometry->set_vertices  ( _corners[0], _corners[1], _corners[2], _corners[3], _corners[4], _corners[5], _corners[6], _corners[7] );

    // retreive infos ffrom octree
    info_traversal  info_traverser;
    _octree->accept ( info_traverser );

    // print out infoss
    info_traverser.print ( std::cout );
    std::cout << "octree size : " << _octree->min() << " - " << _octree->max() << std::endl;
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_octreebased::update_gl_structure ()
  {
    unregister_cuda_structure ();

    _octree_node_arraybuffer->update  ( _nodebuffer.begin(),  _nodebuffer.end());
    _octree_face_arraybuffer->update  ( _facebuffer.begin(),  _facebuffer.end());
    _octree_bbox_arraybuffer->update  ( _bboxbuffer.begin(),  _bboxbuffer.end());
    _octree_limit_arraybuffer->update ( _limitbuffer.begin(), _limitbuffer.end());

    std::size_t nodebufsize  = _nodebuffer.size() * sizeof(gpucast::gl::vec4u);
    std::size_t facebufsize  = _facebuffer.size() * sizeof(gpucast::gl::vec4u);
    std::size_t bboxbufsize  = _bboxbuffer.size() * sizeof(gpucast::gl::vec4f);
    std::size_t limitbufsize = _limitbuffer.size() * sizeof(float);

    std::cout << "Allocating : " << nodebufsize  << " Bytes" << std::endl;
    std::cout << "Allocating : " << facebufsize  << " Bytes" << std::endl;
    std::cout << "Allocating : " << bboxbufsize  << " Bytes" << std::endl;
    std::cout << "Allocating : " << limitbufsize << " Bytes" << std::endl;

    std::cout << "total mem  :" << nodebufsize + facebufsize + bboxbufsize + limitbufsize << " Bytes" << std::endl;


    register_cuda_structure ();
  }

} // namespace gpucast

