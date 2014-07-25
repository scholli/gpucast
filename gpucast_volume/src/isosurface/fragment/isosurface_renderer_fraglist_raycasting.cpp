/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_fraglist_raycasting.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/fragment/isosurface_renderer_fraglist_raycasting.hpp"

// header, system
#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/error.hpp>

//// header, project
#include <gpucast/volume/cuda_map_resources.hpp>


namespace gpucast {

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_fraglist_raycasting::isosurface_renderer_fraglist_raycasting( int argc, char** argv )
    : basetype                      ( argc, argv ),
      _clear_textures_pass          (),
      _quad_pass                    (),
      _intersect_surface_pass       (),
      _fbo                          (),
      _depthattachment              (),
      _colorattachment              (),
      _no_interpolation             (),
      _linear_interpolation         (),
      _initialized_gl               ( false ),
      _cuda_resources_mapped        ( false ),
      _cuda_colorbuffer             ( 0 ),
      _cuda_depthbuffer             ( 0 ),
      _cuda_headpointer             ( 0 ),
      _cuda_fragmentcount           ( 0 ),
      _cuda_indexlist               ( 0 ),
      _cuda_matrixbuffer            ( 0 ),
      _cuda_allocation_grid         ( 0 ),
      _cuda_surface_data_buffer     ( 0 ),
      _cuda_surface_points_buffer   ( 0 ), 
      _cuda_volume_data_buffer      ( 0 ),
      _cuda_volume_points_buffer    ( 0 ),
      _cuda_attribute_data_buffer   ( 0 ),
      _cuda_attribute_points_buffer ( 0 ),
      _cuda_external_texture        ( 0 ), 
      _matrixbuffer                 ()
  {
    _discard_by_minmax   = false;
    _backface_culling    = false;
    _initialized_cuda    = false;
    _render_side_chulls  = true;
    //_background          = gpucast::gl::vec3f(0.0, 0.0, 0.0);
  }


  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_fraglist_raycasting::~isosurface_renderer_fraglist_raycasting()
  { 
    unregister_resource ( &_cuda_colorbuffer );
    unregister_resource ( &_cuda_depthbuffer );
    unregister_resource ( &_cuda_headpointer );
    unregister_resource ( &_cuda_fragmentcount );

    unregister_resource ( &_cuda_indexlist );
    unregister_resource ( &_cuda_matrixbuffer );
    unregister_resource ( &_cuda_allocation_grid );

    unregister_resource ( &_cuda_surface_data_buffer );
    unregister_resource ( &_cuda_surface_points_buffer );
    unregister_resource ( &_cuda_volume_data_buffer );
    unregister_resource ( &_cuda_volume_points_buffer );
    unregister_resource ( &_cuda_attribute_data_buffer );
    unregister_resource ( &_cuda_attribute_points_buffer );

    unregister_resource ( &_cuda_external_texture );
  }

  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                  
  isosurface_renderer_fraglist_raycasting::init ( drawable_ptr const& object,
                                                  std::string const&  attribute_name )
  {
    volume_renderer::set(object, attribute_name);

    _drawable.reset(new drawable_ressource_impl);

    _object_initialized = false;
    _gl_initialized     = false;

    // init with current configuration
    _object_initialized = initialize ( attribute_name );
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */  void                            
  isosurface_renderer_fraglist_raycasting::recompile ()
  {
    fragmentlist_generator::recompile();

    _quad_pass.reset();

    // re-initialize
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */  void                            
  isosurface_renderer_fraglist_raycasting::resize ( int width, int height )
  {
    std::cout << "begin isosurface_renderer_fraglist_raycasting::resize" << std::endl;

    basetype::resize(width, height);

    unregister_resource ( &_cuda_colorbuffer );
    unregister_resource ( &_cuda_depthbuffer );
    unregister_resource ( &_cuda_headpointer );
    unregister_resource ( &_cuda_fragmentcount );

    _init_glresources ();
    _init_cuda        ();

    _colorattachment->teximage ( 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
    _depthattachment->teximage ( 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0); // GL_DEPTH_COMPONENT32 not supported by CUDA4.1
    
    // remap resources
    register_cuda_resources();

    std::cout << "end isosurface_renderer_fraglist_raycasting::resize" << std::endl;
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                            
  isosurface_renderer_fraglist_raycasting::set_external_texture ( std::shared_ptr<gpucast::gl::texture2d> const& texture )
  {
    volume_renderer::set_external_texture(texture);
    unregister_resource(&_cuda_external_texture);
    register_cuda_resources();
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  isosurface_renderer_fraglist_raycasting::draw ()
  {
    _clean_textures();
    
    std::cout << "page: " << pagesize() 
              << "-" << allocation_grid_width() 
              << "| " << allocation_grid_height()<< " ";
              
    gpucast::gl::time_duration time_generation;

    gpucast::gl::timer t;
    t.start();

    // generate fragment lists and sort them
    basetype::draw();
    glFinish();
    t.stop();
 
    time_generation = t.result();

    basetype::readback();

    // compose result
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // make sure everything is initialized
    _init_glresources      ();
    _init_cuda             ();
    _init_platform         ();
    _init_shader           ();

    register_cuda_resources ();

    // intersect surfaces and write result back to imagebuffer
    _update_matrices();

    // execute raycasting
    raycast_fragment_lists();

    // draw final result as a quad
    _draw_result();

   std::cout  << " ms fraggen: " << time_generation.as_seconds() * 1000.0f
              << " ~ #frags : " << _usage_indexbuffer  
              << "      \r";
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_fraglist_raycasting::_init_glresources ()
  {
    if ( _initialized_gl ) return;

    // init GL resources
    _colorattachment.reset      ( new gpucast::gl::texture2d );
    _depthattachment.reset      ( new gpucast::gl::texture2d );
                                  
    _linear_interpolation.reset ( new gpucast::gl::sampler );
                     
    _no_interpolation.reset     ( new gpucast::gl::sampler );
    _no_interpolation->parameter( GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    _no_interpolation->parameter( GL_TEXTURE_MIN_FILTER, GL_NEAREST );

    // allocate memory for matrices
    _matrixbuffer.reset         ( new gpucast::gl::arraybuffer );
    _matrixbuffer->bufferdata   ( 5 * sizeof(gpucast::gl::matrix4f), 0, GL_STATIC_READ ); 

    _initialized_gl = true;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_fraglist_raycasting::_init_platform()
  {
    // lazy init of kernel 
    if ( !_initialized_cuda )
    {
      std::cout << "Initializing CUDA ..." << std::endl;

      _init_cuda();
      _initialized_cuda = true;
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  isosurface_renderer_fraglist_raycasting::_clean_textures ()
  {
    if ( !_fbo )
    {
      _fbo.reset ( new gpucast::gl::framebufferobject );

      _fbo->attach_texture ( *_colorattachment, GL_COLOR_ATTACHMENT0_EXT );
      _fbo->attach_texture ( *_depthattachment, GL_COLOR_ATTACHMENT1_EXT );

      _fbo->bind();
      _fbo->status();
      _fbo->unbind();
    }

    int fbo_bound;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &fbo_bound );

    _fbo->bind();
    glDrawBuffer ( GL_COLOR_ATTACHMENT1_EXT );
    glClearColor ( 1.0f, 1.0f, 1.0f, 1.0f );
    glClear      ( GL_COLOR_BUFFER_BIT );

    glDrawBuffer ( GL_COLOR_ATTACHMENT0_EXT );
    glClearColor ( _background[0], _background[1], _background[2], 1.0f );
    glClear      ( GL_COLOR_BUFFER_BIT );
    _fbo->unbind();

    if ( fbo_bound != 0 ) 
    {
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_bound);
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_fraglist_raycasting::_init_shader ( )
  {
    
    // init fragment generator shaders
    basetype::_init_shader();

    // init glsl programs
    if ( !_quad_pass ) {
      init_program  ( _quad_pass,               "/volumefraglistraycasting/fraglist_raycasting.vert", "/volumefraglistraycasting/draw_from_textures.frag" );
    }

    if ( !_intersect_surface_pass) {
      std::cout << "Initializing Shaders ..." << std::endl;

      init_program  ( _intersect_surface_pass,  "/volumefraglistraycasting/fraglist_raycasting.vert", "/volumefraglistraycasting/fraglist_raycasting.frag" );
    }

    // cuda kernels are linked anyway
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  isosurface_renderer_fraglist_raycasting::register_cuda_resources ()
  {
    // map glresources to kernel resources
    cudaError_t cuda_err = cudaSuccess;

    register_buffer ( &_cuda_surface_data_buffer,     _surface_data_texturebuffer,     cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_surface_points_buffer,   _surface_points_texturebuffer,   cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_volume_data_buffer,      _volume_data_texturebuffer,      cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_volume_points_buffer,    _volume_points_texturebuffer,    cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_attribute_data_buffer,   _attribute_data_texturebuffer,   cudaGraphicsRegisterFlagsReadOnly );
    register_buffer ( &_cuda_attribute_points_buffer, _attribute_points_texturebuffer, cudaGraphicsRegisterFlagsReadOnly );

    if ( !_cuda_colorbuffer )  
      register_image ( &_cuda_colorbuffer,      _colorattachment->id(),   _colorattachment->target(), cudaGraphicsRegisterFlagsSurfaceLoadStore );
    if ( !_cuda_depthbuffer )  
      register_image ( &_cuda_depthbuffer,      _depthattachment->id(),   _depthattachment->target(), cudaGraphicsRegisterFlagsSurfaceLoadStore );\
    if ( !_cuda_headpointer )  
      register_image ( &_cuda_headpointer,      _indextexture->id(),      _indextexture->target(),    cudaGraphicsRegisterFlagsSurfaceLoadStore );
    if ( !_cuda_fragmentcount )  
      register_image ( &_cuda_fragmentcount,    _fragmentcount->id(),     _fragmentcount->target(),   cudaGraphicsRegisterFlagsSurfaceLoadStore );

    if ( _external_color_depth_texture )
      register_image ( &_cuda_external_texture,   _external_color_depth_texture->id(),     _external_color_depth_texture->target(), cudaGraphicsRegisterFlagsSurfaceLoadStore );
    
    register_buffer ( &_cuda_indexlist,       *_indexlist,       cudaGraphicsRegisterFlagsNone );
    register_buffer ( &_cuda_matrixbuffer,    *_matrixbuffer,    cudaGraphicsRegisterFlagsReadOnly ); 
    register_buffer ( &_cuda_allocation_grid, *_allocation_grid, cudaGraphicsRegisterFlagsNone );
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                          
  isosurface_renderer_fraglist_raycasting::unregister_cuda_resources ()
  {
    unregister_resource ( &_cuda_surface_data_buffer );
    unregister_resource ( &_cuda_surface_points_buffer );
    unregister_resource ( &_cuda_volume_data_buffer );
    unregister_resource ( &_cuda_volume_points_buffer );
    unregister_resource ( &_cuda_attribute_data_buffer );
    unregister_resource ( &_cuda_attribute_points_buffer );

    unregister_resource ( &_cuda_colorbuffer );
    unregister_resource ( &_cuda_depthbuffer );
    unregister_resource ( &_cuda_headpointer );
    unregister_resource ( &_cuda_fragmentcount );

    unregister_resource ( &_cuda_external_texture );

    unregister_resource ( &_cuda_indexlist );
    unregister_resource ( &_cuda_matrixbuffer );
    unregister_resource ( &_cuda_allocation_grid );
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  isosurface_renderer_fraglist_raycasting::_update_matrices()
  {
    std::size_t matrix_elements = sizeof(gpucast::gl::matrix4f)/sizeof(gpucast::gl::matrix4f::value_type);
    std::vector<float> matrixbuffer;

    gpucast::gl::matrix4f modelview        = _modelviewmatrix * _modelmatrix;
    gpucast::gl::matrix4f modelviewinverse = gpucast::gl::inverse(modelview);
    gpucast::gl::matrix4f normalmatrix     = modelview.normalmatrix();
    gpucast::gl::matrix4f mvp              = _projectionmatrix * modelview;
    gpucast::gl::matrix4f mvp_inv          = gpucast::gl::inverse(mvp);

    std::copy(&modelview[0],        &modelview[0]         + matrix_elements, std::back_inserter(matrixbuffer));
    std::copy(&modelviewinverse[0], &modelviewinverse[0]  + matrix_elements, std::back_inserter(matrixbuffer));
    std::copy(&mvp[0],              &mvp[0]               + matrix_elements, std::back_inserter(matrixbuffer));
    std::copy(&normalmatrix[0],     &normalmatrix[0]      + matrix_elements, std::back_inserter(matrixbuffer));
    std::copy(&mvp_inv[0],          &mvp_inv[0]           + matrix_elements, std::back_inserter(matrixbuffer));

    try {
      _matrixbuffer->update(matrixbuffer.begin(), matrixbuffer.end());
    } catch ( ... ) {
      std::cerr << "isosurface_renderer_fraglist_raycasting::_update_matrices(): Matrix Buffer Update failed" << std::endl;
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  isosurface_renderer_fraglist_raycasting::_draw_result ()
  {
    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // render quad 
    _quad_pass->begin();
    {
      _quad_pass->set_uniform1i("width", _width);
      _quad_pass->set_uniform1i("height", _height);

      _quad_pass->set_texture2d("colortexture", *_colorattachment, 0);
      //_linear_interpolation->bind(0);
      _no_interpolation->bind(0);
      
      _quad_pass->set_texture2d("depthtexture", *_depthattachment, 1);
      //_linear_interpolation->bind(1);
      _no_interpolation->bind(1);

      _quad->draw();
    }
    _quad_pass->end();

    glDisable(GL_BLEND);
  }


} // namespace gpucast
