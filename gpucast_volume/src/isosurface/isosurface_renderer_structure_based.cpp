/********************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_structure_based.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/isosurface_renderer_structure_based.hpp"

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

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_structure_based::isosurface_renderer_structure_based( int argc, char** argv )
    : volume_renderer               ( argc, argv ),
      _gl_initialized               ( false ),
      _gl_updated                   ( false ),
      _cuda_input_color_depth       ( 0 ),
      _cuda_external_color_depth    ( 0 ),
      _cuda_surface_data_buffer     ( 0 ),
      _cuda_surface_points_buffer   ( 0 ),
      _cuda_volume_data_buffer      ( 0 ),
      _cuda_volume_points_buffer    ( 0 ),
      _cuda_attribute_data_buffer   ( 0 ),
      _cuda_attribute_points_buffer ( 0 ),
      _cuda_matrixbuffer            ( 0 ),
      _cuda_output_color            ( 0 ),
      _cuda_output_depth            ( 0 )
  {}


  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_structure_based::~isosurface_renderer_structure_based()
  {
    unregister_cuda_resources ();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_structure_based::init ( drawable_ptr const& object, std::string const&  attribute_name )
  {
    set ( object, attribute_name );

    _serialize ( attribute_name );

    _gl_updated         = false;

    create_data_structure();
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_structure_based::clear ()
  {
    _object.reset();
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  isosurface_renderer_structure_based::draw ()
  {
    if ( !_object && !_external_color_depth_texture )
    {
      return;
    }

    //gpucast::gl::timer t;
    //glFinish();
    //t.start();

    // make sure everything's initialized
    if ( !_gl_initialized ) 
    {
      _init_gl_resources  ();
      _gl_initialized = true;
    }

    _update_matrixbuffer();
    

    //glFinish();
    //t.stop();
    //gpucast::gl::time_duration t0 = t.result();
    //t.start();

    //glFinish();
    //t.stop();
    //gpucast::gl::time_duration t1 = t.result();
    //t.start();

        // clean up fbo
    _clean_textures        ();

    //glFinish();
    //t.stop();
    //gpucast::gl::time_duration t2 = t.result();
    //t.start();

    if ( !_gl_updated ) 
    {
      update_gl_resources ();
      update_gl_structure ();
      _gl_updated = true;
    }

    //glFinish();
    //t.stop();
    //gpucast::gl::time_duration t3 = t.result();
    //t.start();

    // draw bbox for ray generation
    raygeneration();

    //glFinish();
    //t.stop();
    //gpucast::gl::time_duration t4 = t.result();
    //t.start();

    // ray cast 
    if ( !_object )
    {
      if ( _cuda_output_color && _cuda_output_depth && _cuda_external_color_depth ) {
        invoke_external_passthrough ( _width, _height, _cuda_output_color, _cuda_output_depth, _cuda_external_color_depth );
      }
    } else {
      // get octree dimensions
      float attribute_minimum = _global_attribute_bounds.minimum();
      float attribute_maximum = _global_attribute_bounds.maximum();
      float attribute_range   = attribute_maximum - attribute_minimum;

      renderconfig config;
      config.adaptive_sampling  = _adaptive_sampling;
      config.attrib_min         = attribute_minimum;
      config.attrib_max         = attribute_maximum;
      config.backface_culling   = _backface_culling;
      config.boundary_opacity   = _surface_transparency;
      config.farplane           = _farplane;
      config.nearplane          = _nearplane;
      config.isosurface_opacity = _isosurface_transparency;
      config.isovalue           = attribute_minimum + attribute_range * _relative_isovalue;
      config.max_binary_steps   = _max_binary_searches;
      config.max_octree_depth   = _max_octree_depth;
      config.newton_epsilon     = _newton_epsilon;
      config.newton_iterations  = _newton_iterations;
      config.bbox_min[0]        = _bbox.min[0];
      config.bbox_min[1]        = _bbox.min[1];
      config.bbox_min[2]        = _bbox.min[2];
      config.bbox_max[0]        = _bbox.max[0];
      config.bbox_max[1]        = _bbox.max[1];
      config.bbox_max[2]        = _bbox.max[2];
      config.screenspace_newton_error = _screenspace_newton_error;
      config.volume_info_offset = volume_renderer::volume_data_header;
      config.steplength_min     = _min_sample_distance;
      config.steplength_max     = _max_sample_distance;
      config.steplength_scale   = _adaptive_sample_scale;
      config.width              = _width;
      config.height             = _height;
      config.show_isosides      = _visualization_props.show_isosides;

      invoke_ray_casting_kernel ( config );
    }

    //glFinish();
    //t.stop();
    //gpucast::gl::time_duration t5 = t.result();
    //t.start();

    // map result to screen
    _draw_result();

    //glFinish();
    //t.stop();
    //gpucast::gl::time_duration t6 = t.result();
    //t.start();
#if 0 
    std::cout << "t0: " << t0 << std::endl << 
                 "t1: " << t1 << std::endl << 
                 "t2: " << t2 << std::endl << 
                 "t3: " << t3 << std::endl << 
                 "t4: " << t4 << std::endl << 
                 "t5: " << t5 << std::endl << 
                 "t6: " << t6 << std::endl;
#endif
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_structure_based::raygeneration ()
  {
    // store old state
    GLint cull_face, front_face, cull_mode, fbo;
    glGetIntegerv ( GL_FRAMEBUFFER_BINDING, &fbo );
    glGetIntegerv ( GL_CULL_FACE,           &cull_face );
    glGetIntegerv ( GL_FRONT_FACE,          &front_face );
    glGetIntegerv ( GL_CULL_FACE_MODE,      &cull_mode );

    // apply necessary state
    glEnable      ( GL_CULL_FACE ); // front_faces are culled for ray generation
    glFrontFace   ( GL_CW );
    glCullFace    ( GL_FRONT );

    _raygeneration_fbo->bind(); // draw into fbo
    glClearColor ( _background[0], _background[1], _background[2], _background[3]);
    glClearDepth ( 1.0f );
    glDepthFunc ( GL_LESS );

    //glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    {
      _raygeneration_program->begin();
      {
        _raygeneration_program->set_uniform_matrix4fv ( "modelviewprojectionmatrix", 1, false, &_modelviewprojectionmatrix[0] );
        _raygeneration_geometry->draw();
      }
      _raygeneration_program->end();
    }
    _raygeneration_fbo->unbind();
    //glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL );
    // restore old state
    if ( fbo )                   glBindFramebuffer ( GL_FRAMEBUFFER_EXT, fbo );
    if ( !cull_face )            glDisable         ( GL_CULL_FACE );
    if ( cull_mode != GL_FRONT ) glCullFace        ( cull_mode );
    if ( front_face != GL_CW )  glFrontFace       ( front_face );
  }


  /////////////////////////////////////////////////////////////////////////////
  void                      
  isosurface_renderer_structure_based::transform ( gpucast::math::matrix4f const& m )
  {
    _modelmatrix = m;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_structure_based::recompile ()
  {
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  void              
  isosurface_renderer_structure_based::resize ( int w, int h )
  {
    // base functionality
    volume_renderer::resize(w, h);

    // unregister CUDA resources that depend on screensize
    unregister_resource ( &_cuda_output_color );
    unregister_resource ( &_cuda_output_depth );
    unregister_resource ( &_cuda_input_color_depth );
    unregister_resource ( &_cuda_external_color_depth );

    // make sure GL resources are initialized
    if ( !_gl_initialized ) 
    {
      _init_gl_resources  (); 
      _gl_initialized = true;
    }

    _raygeneration_color->teximage ( 0, GL_RGBA32F,              _width, _height, 0, GL_RGBA, GL_FLOAT, 0);
    _raygeneration_depth->set      ( GL_DEPTH24_STENCIL8_EXT, _width, _height);

    // allocate GL textures that depend on screensize
    _color_texture->teximage ( 0, GL_RGBA32F, _width, _height, 0, GL_RGBA, GL_FLOAT, 0);
    _depth_texture->teximage ( 0, GL_R32F, _width, _height,    0, GL_RED, GL_FLOAT, 0); // GL_DEPTH_COMPONENT32 not supported by CUDA4.1

    // remap resources
    register_cuda_resources();
  }


    /////////////////////////////////////////////////////////////////////////////
  void                                          
  isosurface_renderer_structure_based::unregister_cuda_resources ()
  {
    unregister_resource ( &_cuda_surface_data_buffer );
    unregister_resource ( &_cuda_surface_points_buffer );
    unregister_resource ( &_cuda_volume_data_buffer );
    unregister_resource ( &_cuda_volume_points_buffer );
    unregister_resource ( &_cuda_attribute_data_buffer );
    unregister_resource ( &_cuda_attribute_points_buffer );

    unregister_resource ( &_cuda_matrixbuffer );

    unregister_resource ( &_cuda_output_color );
    unregister_resource ( &_cuda_output_depth );

    unregister_resource ( &_cuda_external_color_depth );
    unregister_resource ( &_cuda_input_color_depth );
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                            
  isosurface_renderer_structure_based::set_external_texture ( std::shared_ptr<gpucast::gl::texture2d> const& texture )
  {
    volume_renderer::set_external_texture(texture);
    unregister_resource(&_cuda_external_color_depth);
    register_cuda_resources();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_structure_based::write ( std::ostream& os ) const
  {
    os.write ( reinterpret_cast<char const*> (&_bbox.min[0]), sizeof(node::boundingbox_type::point_type));
    os.write ( reinterpret_cast<char const*> (&_bbox.max[0]), sizeof(node::boundingbox_type::point_type));

    gpucast::write (os, _surface_data);
    gpucast::write (os, _surface_points);
    gpucast::write (os, _volume_data);
    gpucast::write (os, _volume_points);
    gpucast::write (os, _attribute_data);
    gpucast::write (os, _attribute_points);

    gpucast::write (os, _corners);
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  isosurface_renderer_structure_based::read ( std::istream& is )
  {
    if ( !_gl_initialized ) 
    {
      _init_gl_resources  ();
      _gl_initialized = true;
    }

    is.read ( reinterpret_cast<char*> (&_bbox.min[0]), sizeof(node::boundingbox_type::point_type));
    is.read ( reinterpret_cast<char*> (&_bbox.max[0]), sizeof(node::boundingbox_type::point_type));

    gpucast::read (is, _surface_data);
    gpucast::read (is, _surface_points);
    gpucast::read (is, _volume_data);
    gpucast::read (is, _volume_points);
    gpucast::read (is, _attribute_data);
    gpucast::read (is, _attribute_points);

    // copy to GPU
    gpucast::read (is, _corners);
    _raygeneration_geometry->set_vertices  ( _corners[0], _corners[1], _corners[2], _corners[3], _corners[4], _corners[5], _corners[6], _corners[7] );
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_structure_based::_init_shader ()
  {
    volume_renderer::_init_shader();

    init_program ( _raygeneration_program,  "/octree/raygeneration.vert",                         "/octree/raygeneration.frag" );
    init_program ( _map_quad_program,       "/volumefraglistraycasting/fraglist_raycasting.vert", "/volumefraglistraycasting/draw_from_textures.frag" );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_structure_based::_init_gl_resources ()
  {
    _init_cuda                                ();
    _init_shader                              ();

    float initial_depth = 1.0f;

    init_structure();

    _surface_data_arraybuffer.reset           ( new gpucast::gl::arraybuffer );
    _surface_points_arraybuffer.reset         ( new gpucast::gl::arraybuffer );
    _volume_data_arraybuffer.reset            ( new gpucast::gl::arraybuffer );
    _volume_points_arraybuffer.reset          ( new gpucast::gl::arraybuffer );
    _attribute_data_arraybuffer.reset         ( new gpucast::gl::arraybuffer );
    _attribute_points_arraybuffer.reset       ( new gpucast::gl::arraybuffer );

    _matrix_arraybuffer.reset                 ( new gpucast::gl::arraybuffer );
    _matrix_arraybuffer->bufferdata           ( 5 * sizeof(gpucast::math::matrix4f), 0, GL_STATIC_DRAW );

    _raygeneration_geometry.reset             ( new gpucast::gl::cube );
    
    _raygeneration_color.reset                ( new gpucast::gl::texture2d );
    _raygeneration_depth.reset                ( new gpucast::gl::renderbuffer );
    _raygeneration_color->teximage            ( 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, &_background[0] );
    _raygeneration_depth->set                 ( GL_DEPTH_COMPONENT32F, 1, 1 );

    _raygeneration_fbo.reset                  ( new gpucast::gl::framebufferobject );
    _raygeneration_fbo->attach_texture        ( *_raygeneration_color, GL_COLOR_ATTACHMENT0_EXT );
    _raygeneration_fbo->attach_renderbuffer   ( *_raygeneration_depth, GL_DEPTH_ATTACHMENT_EXT );

    _raygeneration_fbo->bind();
    _raygeneration_fbo->status();
    _raygeneration_fbo->unbind();

    _color_texture.reset                      ( new gpucast::gl::texture2d );
    _depth_texture.reset                      ( new gpucast::gl::texture2d );

    _color_texture->teximage                  ( 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, &_background[0] );
    _depth_texture->teximage                  ( 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, &initial_depth );

    _output_fbo.reset                         ( new gpucast::gl::framebufferobject );
    _output_fbo->attach_texture               ( *_color_texture, GL_COLOR_ATTACHMENT0_EXT );
    _output_fbo->attach_texture               ( *_depth_texture, GL_COLOR_ATTACHMENT1_EXT );

    _output_fbo->bind();
    _output_fbo->status();
    _output_fbo->unbind();

    _nearest_interpolation.reset              ( new gpucast::gl::sampler );
    _nearest_interpolation->parameter         ( GL_TEXTURE_WRAP_S, GL_REPEAT );
    _nearest_interpolation->parameter         ( GL_TEXTURE_WRAP_T, GL_REPEAT );
    _nearest_interpolation->parameter         ( GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    _nearest_interpolation->parameter         ( GL_TEXTURE_MIN_FILTER, GL_NEAREST );

    _map_quad.reset                           ( new gpucast::gl::plane(0, -1, 1) );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_structure_based::_update_matrixbuffer()
  {
    unsigned const elements  = 16;
    unsigned const nmatrices = 5;

    std::vector<float> matrixbuffer ( elements * nmatrices );

    gpucast::math::matrix4f modelviewmatrix                    = _modelviewmatrix;
    gpucast::math::matrix4f modelviewprojectionmatrix          = _modelviewprojectionmatrix;
    gpucast::math::matrix4f modelviewinverse                   = gpucast::math::inverse(_modelviewmatrix);
    gpucast::math::matrix4f modelviewprojectionmatrixinverse   = _modelviewprojectionmatrixinverse;
    gpucast::math::matrix4f normalmatrix                       = _modelviewmatrix.normalmatrix();

    auto ins_pos = matrixbuffer.begin();
    ins_pos = std::copy(&modelviewmatrix[0],                   &modelviewmatrix[16],                  ins_pos );
    ins_pos = std::copy(&modelviewprojectionmatrix[0],         &modelviewprojectionmatrix[16],        ins_pos );
    ins_pos = std::copy(&modelviewinverse[0],                  &modelviewinverse[16],                 ins_pos );
    ins_pos = std::copy(&normalmatrix[0],                      &normalmatrix[16],                     ins_pos );
    ins_pos = std::copy(&modelviewprojectionmatrixinverse[0],  &modelviewprojectionmatrixinverse[16], ins_pos );
    
    _matrix_arraybuffer->update ( matrixbuffer.begin(), matrixbuffer.end() );
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_structure_based::register_cuda_resources ()
  {
    register_buffer ( &_cuda_surface_data_buffer,     *_surface_data_arraybuffer,    cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_surface_points_buffer,   *_surface_points_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_volume_data_buffer,      *_volume_data_arraybuffer,     cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_volume_points_buffer,    *_volume_points_arraybuffer,   cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_attribute_data_buffer,   *_attribute_data_arraybuffer,  cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_attribute_points_buffer, *_attribute_points_arraybuffer,cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_matrixbuffer,            *_matrix_arraybuffer,          cudaGraphicsRegisterFlagsReadOnly );

    register_image    ( &_cuda_input_color_depth, _raygeneration_color->id(), _raygeneration_color->target(), cudaGraphicsRegisterFlagsSurfaceLoadStore );
    register_image    ( &_cuda_output_color,      _color_texture->id(),       _color_texture->target(),       cudaGraphicsRegisterFlagsSurfaceLoadStore );
    register_image    ( &_cuda_output_depth,      _depth_texture->id(),       _depth_texture->target(),       cudaGraphicsRegisterFlagsSurfaceLoadStore );

    if ( _external_color_depth_texture ) {
      register_image ( &_cuda_external_color_depth, _external_color_depth_texture->id(), _external_color_depth_texture->target(), cudaGraphicsRegisterFlagsSurfaceLoadStore );
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  isosurface_renderer_structure_based::_draw_result ()
  {
    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // render quad 
    _map_quad_program->begin();
    {
      _map_quad_program->set_uniform1i("width", _width);
      _map_quad_program->set_uniform1i("height", _height);

      //_map_quad_program->set_texture2d("colortexture", *_color_texture, 0);
      _map_quad_program->set_texture2d("colortexture", *_color_texture, 0);
      //_linear_interpolation->bind(0);
      _nearest_interpolation->bind(0);
      
      _map_quad_program->set_texture2d("depthtexture", *_depth_texture, 1);
      //_linear_interpolation->bind(1);
      _nearest_interpolation->bind(1);

      _map_quad->draw();
    }
    _map_quad_program->end();

    glDisable(GL_BLEND);
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_structure_based::_clean_textures ()
  {
    int fbo_bound;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &fbo_bound );

    _output_fbo->bind();
    glDrawBuffer ( GL_COLOR_ATTACHMENT1_EXT );
    glClearColor ( 1.0f, 1.0f, 1.0f, 1.0f );
    glClear      ( GL_COLOR_BUFFER_BIT );

    glDrawBuffer ( GL_COLOR_ATTACHMENT0_EXT );
    glClearColor ( _background[0], _background[1], _background[2], 1.0f );
    glClear      ( GL_COLOR_BUFFER_BIT );
    _output_fbo->unbind();

    if ( fbo_bound != 0 ) 
    {
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_bound);
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_structure_based::update_gl_resources ()
  {
    unregister_cuda_resources ();

    _surface_data_arraybuffer->update    ( _surface_data.begin(), _surface_data.end());
    _surface_points_arraybuffer->update  ( _surface_points.begin(), _surface_points.end());
    _volume_data_arraybuffer->update     ( _volume_data.begin(),  _volume_data.end());
    _volume_points_arraybuffer->update   ( _volume_points.begin(),  _volume_points.end());
    _attribute_data_arraybuffer->update  ( _attribute_data.begin(),  _attribute_data.end());
    _attribute_points_arraybuffer->update( _attribute_points.begin(),  _attribute_points.end());

    register_cuda_resources ();
  }

////////////////////////////////////////////////////////////////////////////////
  bool 
  isosurface_renderer_structure_based::_serialize ( std::string const& attribute_name )
  {
    if ( _surface_data.empty()     ) _surface_data.resize(_empty_slots); 
    if ( _surface_points.empty()   ) _surface_points.resize(_empty_slots);
    if ( _volume_data.empty()      ) _volume_data.resize(_empty_slots);
    if ( _volume_points.empty()    ) _volume_points.resize(_empty_slots); 
    if ( _attribute_data.empty()   ) _attribute_data.resize(_empty_slots); 
    if ( _attribute_points.empty() ) _attribute_points.resize(_empty_slots); 
  
    // initialize helper structures
    std::vector<deferred_surface_header_write>  deferred_surface_jobs;

    std::map<unsigned, unsigned>                uniqueid_volume_index_map; 
    std::map<unsigned, unsigned>                uniqueid_attribute_buffer_index_map; 

    // progress stream output
    int cnt = 0;
    int one_percent = std::max(1, int(_object->size())/100);

    // iterate all volumes and compute outer surface hulls
    for ( beziervolumeobject::const_iterator v = _object->begin(); v != _object->end(); ++v )
    {
      ++cnt;
    
      if ( cnt % one_percent == 0 )
      {
        std::cout << "Processing volumes : " << float(cnt*100)/_object->size() << "%\r";
      }

      // store information about volume at first index
      unsigned volume_data_id   = unsigned(_volume_data.size());
      unsigned volume_points_id = unsigned(_volume_points.size());

      uniqueid_volume_index_map.insert ( std::make_pair( v->id(), unsigned(volume_data_id) ) );

      // push_back proxies for volume info indices
      _volume_data.resize ( _volume_data.size() + volume_renderer::volume_data_header );

      // transform volume controlpoints to hyperspace and copy to volume-buffer
      std::transform (v->begin(), v->end(), std::back_inserter(_volume_points), hyperspace_adapter_3D_to_4D<beziervolume::point_type, gpucast::math::vec4f>());

      // write attached data into buffer and store related index
      unsigned attribute_data_id   = 0;
      unsigned attribute_points_id = 0;

      beziervolume::attribute_volume_type const& attrib_vol = (*v)[attribute_name];

      // store current size of container
      attribute_data_id   = unsigned(_attribute_data.size());
      attribute_points_id = unsigned(_attribute_points.size());

      uniqueid_attribute_buffer_index_map.insert ( std::make_pair( v->id(), unsigned(attribute_data_id) ) );

      // write information about attribute
      float attrib_min;
      float attrib_max;

      attrib_min = attrib_vol.bbox().min[beziervolume::point_type::x];
      attrib_max = attrib_vol.bbox().max[beziervolume::point_type::x];

      _attribute_data.push_back ( gpucast::math::vec4f ( attrib_min, attrib_max, bit_cast<unsigned, float>(attribute_points_id), 0) );

      // write attribute's control points into buffer
      std::transform ( attrib_vol.begin(), 
                       attrib_vol.end(), 
                       std::back_inserter(_attribute_points), 
                         [] ( beziervolume::attribute_type const& p ) 
                         { 
                           return gpucast::math::vec2f(p.as_homogenous()[0], p.as_homogenous()[1]); 
                         } 
                      );

      // create actual proxy geometry
      _create_proxy_geometry ( *v, deferred_surface_jobs );

      gpucast::math::vec3u order ( unsigned(v->order_u()), unsigned(v->order_v()), unsigned(v->order_w()));

      // write per volume header info into volume data buffer
      _volume_data[volume_data_id    ] = gpucast::math::vec4f(bit_cast<unsigned,float>(volume_points_id), bit_cast<unsigned,float>(v->id()),   bit_cast<unsigned,float>(attribute_data_id),  float(v->bbox().size().abs()) );
      _volume_data[volume_data_id + 1] = gpucast::math::vec4f(bit_cast<unsigned,float>(order[0]),         bit_cast<unsigned,float>(order[1]), bit_cast<unsigned,float>(order[2]),            0.0f);
      _volume_data[volume_data_id + 2] = gpucast::math::vec4f(float(v->uvwmin_local()[0]),   float(v->uvwmin_local()[1]),  float(v->uvwmin_local()[2]),  0.0f);
      _volume_data[volume_data_id + 3] = gpucast::math::vec4f(float(v->uvwmax_local()[0]),   float(v->uvwmax_local()[1]),  float(v->uvwmax_local()[2]),  0.0f);
      _volume_data[volume_data_id + 4] = gpucast::math::vec4f(float(v->uvwmin_global()[0]),  float(v->uvwmin_global()[1]), float(v->uvwmin_global()[2]), 0.0f);
      _volume_data[volume_data_id + 5] = gpucast::math::vec4f(float(v->uvwmax_global()[0]),  float(v->uvwmax_global()[1]), float(v->uvwmax_global()[2]), 0.0f);
      _volume_data[volume_data_id + 6] = gpucast::math::vec4f(float(v->bbox().min[0]),       float(v->bbox().min[1]),      float(v->bbox().min[2]),      0.0);
      _volume_data[volume_data_id + 7] = gpucast::math::vec4f(float(v->bbox().max[0]),       float(v->bbox().max[1]),      float(v->bbox().max[2]),      0.0);
    }

    // do deferred jobs
    cnt = 0;
    one_percent = std::max(1, int(deferred_surface_jobs.size())/100);
    std::cout << std::endl;

    for ( auto j = deferred_surface_jobs.begin(); j != deferred_surface_jobs.end(); ++j )
    {
      ++cnt;
      if ( cnt % one_percent == 0 )
      {
        std::cout << "Processing surfaces : " << float(cnt*100)/deferred_surface_jobs.size() << "\r";
      }

      deferred_surface_header_write const& job = *j;

      unsigned volume_data_id             = uniqueid_volume_index_map.at                ( job.volume_uid );
      unsigned attribute_data_id          = uniqueid_attribute_buffer_index_map.at      ( job.volume_uid );
      unsigned adjacent_volume_data_id    = ( uniqueid_volume_index_map.count           ( job.adjacent_volume_uid ) && !job.outer_face) ? uniqueid_volume_index_map.at           ( job.adjacent_volume_uid ) : 0;
      unsigned adjacent_attribute_data_id = ( uniqueid_attribute_buffer_index_map.count ( job.adjacent_volume_uid ) && !job.outer_face) ? uniqueid_attribute_buffer_index_map.at ( job.adjacent_volume_uid ) : 0;

      float attrib_min = _attribute_data[attribute_data_id][0];
      float attrib_max = _attribute_data[attribute_data_id][1];

      if ( adjacent_attribute_data_id != 0 )
      {
        attrib_min = std::min ( attrib_min, _attribute_data[adjacent_attribute_data_id][0]);
        attrib_max = std::max ( attrib_max, _attribute_data[adjacent_attribute_data_id][1]);
      }
      
      /*
      gpucast::math::vec4u surface_info0 ( job.surface_uid,
                                  job.volume_uid,
                                  volume_data_id, 
                                  attribute_data_id );*/
      gpucast::math::vec4u surface_info0 ( bit_cast<float,unsigned>(attrib_min),
                                  bit_cast<float,unsigned>(attrib_max),
                                  volume_data_id, 
                                  attribute_data_id );

      gpucast::math::vec4u surface_info1 ( job.outer_cell, 
                                  job.surface_points_id, 
                                  adjacent_volume_data_id,
                                  adjacent_attribute_data_id );

      gpucast::math::vec4u surface_info2 ( job.outer_face, 
                                  job.order_u,
                                  job.order_v,
                                  job.surface_type );
                                                                    
      gpucast::math::vec4u surface_info3;

      switch ( job.surface_type ) 
      {
        case beziervolume::boundary_t::umin : surface_info3 = gpucast::math::vec4u(1, 2, 0, 0); break;
        case beziervolume::boundary_t::umax : surface_info3 = gpucast::math::vec4u(1, 2, 0, 1); break;
        case beziervolume::boundary_t::vmin : surface_info3 = gpucast::math::vec4u(0, 2, 1, 0); break;
        case beziervolume::boundary_t::vmax : surface_info3 = gpucast::math::vec4u(0, 2, 1, 1); break;
        case beziervolume::boundary_t::wmin : surface_info3 = gpucast::math::vec4u(0, 1, 2, 0); break;
        case beziervolume::boundary_t::wmax : surface_info3 = gpucast::math::vec4u(0, 1, 2, 1); break;
      };

      _surface_data[job.surface_data_id  ] = surface_info0;
      _surface_data[job.surface_data_id+1] = surface_info1;
      _surface_data[job.surface_data_id+2] = surface_info2;
      _surface_data[job.surface_data_id+3] = surface_info3;
    } 

    std::cout << std::endl;

    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_structure_based::_create_proxy_geometry ( beziervolume const& v,
                                                                 std::vector<deferred_surface_header_write>& deferred_jobs )
  {
    // store surface data base index
    for (unsigned surface = beziervolume::boundary_t::umin; surface != beziervolume::boundary_t::count; ++surface) 
    {
      // create proxy geometry for all outer surfaces and inner minimum surfaces -> avoid multiple proxies for transition face
      if (  v.is_outer()[surface] ||                      // 
            surface == beziervolume::boundary_t::umin ||  // all inner beziersurfaces of uvw-min type
            surface == beziervolume::boundary_t::vmin ||
            surface == beziervolume::boundary_t::wmin
         )
      {
        unsigned surface_data_id   = unsigned ( _surface_data.size() );
        unsigned surface_points_id = unsigned ( _surface_points.size() );

        gpucast::math::beziersurface<gpucast::math::vec3d> outer_face;
        switch ( surface )
        {
          case beziervolume::boundary_t::umin : outer_face = v.slice<gpucast::math::vec3d> (beziervolume::point_type::x, 0); break;
          case beziervolume::boundary_t::umax : outer_face = v.slice<gpucast::math::vec3d> (beziervolume::point_type::x, v.degree_u() ); break;
          case beziervolume::boundary_t::vmin : outer_face = v.slice<gpucast::math::vec3d> (beziervolume::point_type::y, 0 ); break;
          case beziervolume::boundary_t::vmax : outer_face = v.slice<gpucast::math::vec3d> (beziervolume::point_type::y, v.degree_v() ); break;
          case beziervolume::boundary_t::wmin : outer_face = v.slice<gpucast::math::vec3d> (beziervolume::point_type::z, 0 ); break;
          case beziervolume::boundary_t::wmax : outer_face = v.slice<gpucast::math::vec3d> (beziervolume::point_type::z, v.degree_w() ); break;
        }

        // write header of surface
        _surface_data.resize(_surface_data.size() + volume_renderer::surface_data_header);

        // write control points of surface
        std::transform (outer_face.begin(), outer_face.end(), std::back_inserter(_surface_points), hyperspace_adapter_3D_to_4D<gpucast::math::point3d, gpucast::math::vec4f>());

        // deferred fill job, because not all neighbor volumes are stored yet -> not all buffer id's are known
        deferred_surface_header_write   job;

        job.order_u                  = outer_face.order_u();
        job.order_v                  = outer_face.order_v();
        job.outer_face               = v.is_outer()[surface];
        job.outer_cell               = std::accumulate(v.is_outer().begin(), v.is_outer().end(), 0) > 0;
        job.surface_type             = surface;
        job.surface_uid              = v.surface_ids()[surface];
        job.surface_data_id          = surface_data_id;
        job.surface_points_id        = surface_points_id;
        job.volume_uid               = v.id();
        job.adjacent_volume_uid      = v.neighbor_ids()[surface];
        
        deferred_jobs.push_back(job);
      }
    }
  }


  ///////////////////////////////////////////////////////////////////////////////
  void              
  isosurface_renderer_structure_based::_extract_faces ( std::vector<face_ptr>& faces )
  {
    std::size_t i = _empty_slots;

    while ( i < _surface_data.size() )
    {
      face_ptr f ( new face );
      f->surface_id = unsigned(i);

      gpucast::math::vec4u face_info0 = _surface_data[i  ];
      gpucast::math::vec4u face_info1 = _surface_data[i+1];
      gpucast::math::vec4u face_info2 = _surface_data[i+2];

      unsigned order_u          = face_info2[1];
      unsigned order_v          = face_info2[2];
      unsigned surface_point_id = face_info1[1];
      std::size_t npoints = order_u * order_v; 

      auto fst_point = _surface_points.begin(); 
      std::advance(fst_point, surface_point_id);
      auto lst_point = fst_point; 
      std::advance(lst_point, npoints);

      gpucast::math::pointmesh3d<gpucast::math::point3f> mesh (fst_point, lst_point, order_u, order_v, 1);
      f->obb    = gpucast::math::oriented_boundingbox<gpucast::math::point3f> (mesh, gpucast::math::partial_derivative_policy<gpucast::math::point3f>());
      if ( !f->obb.valid() )
        f->obb = gpucast::math::oriented_boundingbox<gpucast::math::point3f> (mesh, gpucast::math::greedy_policy<gpucast::math::point3f>(100));

      f->outer  = face_info2[0] != 0;
    
      // determine attached attribute range
      //gpucast::volume_renderer::attributebuffer_map const& attribute_buffer = r->attribute_data();
      std::size_t attribute_id0 = face_info0[3];
      std::size_t attribute_id1 = face_info1[3];

      // use first component
      f->attribute_range = gpucast::math::interval<float>(_attribute_data[attribute_id0][0], _attribute_data[attribute_id0][1]);

      // if face has an adjacent volume extend attribute range
      if ( attribute_id1 != 0 ) 
      {
        gpucast::math::interval<float> adjacent_range (_attribute_data[attribute_id1][0], _attribute_data[attribute_id1][1]);
        f->attribute_range.merge ( adjacent_range );
      }

      // store face
      faces.push_back(f);

      // iterate to next face
      i += volume_renderer::surface_data_header;
    }
  }


} // namespace gpucast

