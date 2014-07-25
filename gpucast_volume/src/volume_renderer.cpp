/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#if WIN32
  #pragma warning(disable: 4996)
#endif

// header i/f
#include "gpucast/volume/volume_renderer.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// header, system
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>

#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/transferfunction.hpp>

// header, project
#include <gpucast/volume/nurbsvolumeobject.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/volume/uid.hpp>

namespace gpucast {

  // Beginning of GPU Architecture definitions
  int _ConvertSMVer2Cores_local(int major, int minor)
  {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
      int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
      int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    { { 0x10, 8 },
    { 0x11, 8 },
    { 0x12, 8 },
    { 0x13, 8 },
    { 0x20, 32 },
    { 0x21, 48 },
    { -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
      if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
        return nGpuArchCoresPerSM[index].Cores;
      }
      index++;
    }
    return -1;
  }

  /////////////////////////////////////////////////////////////////////////////
  volume_renderer::volume_renderer( int argc, char** argv )
    : renderer                  (),
      _object                   (),
      _relative_isovalue        ( 1.0f ),
      _screenspace_newton_error ( false ),
      _newton_iterations        ( 6 ),
      _newton_epsilon           ( 0.001f ),
      _adaptive_sampling        ( true ),
      _min_sample_distance      ( 0.01f ),
      _max_sample_distance      ( 0.2f ), 
      _adaptive_sample_scale    ( 0.5f ), 
      _max_octree_depth         ( 16 ),
      _max_volumes_per_node     ( 64 ),
      _max_binary_searches      ( 8 ),
      _surface_transparency     ( 0.3f ),
      _isosurface_transparency  ( 0.5f ),
      _backface_culling         ( true ),
      _global_attribute_bounds  ( 0.0f, 1.0f ),
      _base_program             (),
      _transfertexture          (),
      _external_color_depth_texture (),
      _visualization_props      ()
  {
    projectionmatrix(gpucast::gl::frustum(-1.0f, 1.0f, -1.0f, 1.0f, _nearplane, _farplane));

    _init();
  }


  /////////////////////////////////////////////////////////////////////////////
  volume_renderer::~volume_renderer()
  {}


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  volume_renderer::recompile ()
  {
    _base_program.reset();
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void 
  volume_renderer::set ( drawable_ptr const& object, std::string const& attribute_name)
  {
    _object = object;
    _global_attribute_bounds = attribute_interval(object->parent()->bbox(attribute_name).min[0],
                                                  object->parent()->bbox(attribute_name).max[0]);
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::clear ()
  {
    _object.reset();
  }


  /////////////////////////////////////////////////////////////////////////////
  volume_renderer::attribute_interval const&     
  volume_renderer::get_attributebounds ( ) const
  {
    return _global_attribute_bounds;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                  
  volume_renderer::set_attributebounds ( volume_renderer::attribute_interval const& m )
  {
    _global_attribute_bounds = m;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::newton_iterations ( unsigned n )
  {
    _newton_iterations = n;
  }


  /////////////////////////////////////////////////////////////////////////////
  unsigned
  volume_renderer::newton_iterations ( ) const
  {
    return _newton_iterations;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::newton_epsilon ( float e )
  {
    _newton_epsilon = e;
  }


  /////////////////////////////////////////////////////////////////////////////
  float
  volume_renderer::newton_epsilon ( ) const
  {
    return _newton_epsilon;
  }


  /////////////////////////////////////////////////////////////////////////////
  float
  volume_renderer::relative_isovalue () const
  {
    return _relative_isovalue;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::relative_isovalue ( float t )
  {
    _relative_isovalue = t;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool
  volume_renderer::adaptive_sampling () const
  {
    return _adaptive_sampling;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::adaptive_sampling (bool s )
  {
    _adaptive_sampling = s;
  }

  /////////////////////////////////////////////////////////////////////////////
  bool                          
  volume_renderer::screenspace_newton_error () const
  {
    return _screenspace_newton_error;
  }

  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                  
  volume_renderer::screenspace_newton_error ( bool enable )
  {
    _screenspace_newton_error = enable;
  }


  /////////////////////////////////////////////////////////////////////////////
  float
  volume_renderer::min_sample_distance () const
  {
    return _min_sample_distance;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::min_sample_distance (float s )
  {
    _min_sample_distance = s;
  }
  

  /////////////////////////////////////////////////////////////////////////////
  float
  volume_renderer::max_sample_distance () const
  {
    return _max_sample_distance;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::max_sample_distance (float s )
  {
    _max_sample_distance = s;
  }


  /////////////////////////////////////////////////////////////////////////////
  float
  volume_renderer::adaptive_sample_scale () const
  {
    return _adaptive_sample_scale;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::adaptive_sample_scale (float s )
  {
    _adaptive_sample_scale = s;
  }


  /////////////////////////////////////////////////////////////////////////////
  unsigned                            
  volume_renderer::max_octree_depth () const
  {
    return _max_octree_depth;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  volume_renderer::max_octree_depth ( unsigned i )
  {
    _max_octree_depth = std::max(0U, i);
  }


  /////////////////////////////////////////////////////////////////////////////
  unsigned                            
  volume_renderer::max_volumes_per_node () const
  {
    return _max_volumes_per_node;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  volume_renderer::max_volumes_per_node ( unsigned i )
  {
    _max_volumes_per_node = std::max(4U, i);
  }


  /////////////////////////////////////////////////////////////////////////////
  unsigned                        
  volume_renderer::max_binary_searches () const
  {
    return _max_binary_searches;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                            
  volume_renderer::max_binary_searches ( unsigned i )
  {
    _max_binary_searches = i;
  }


  /////////////////////////////////////////////////////////////////////////////
  float                         
  volume_renderer::surface_opacity () const
  {
    return _surface_transparency;
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                  
  volume_renderer::surface_opacity ( float opacity )
  {
    _surface_transparency = opacity;
  }

  
  /////////////////////////////////////////////////////////////////////////////
  float                         
  volume_renderer::isosurface_opacity () const
  {
    return _isosurface_transparency;
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void                  
  volume_renderer::isosurface_opacity ( float opacity )
  {
    _isosurface_transparency = opacity;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool                          
  volume_renderer::backface_culling () const
  {
    return _backface_culling;
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */  void                  
  volume_renderer::backface_culling ( bool enable )
  {
    _backface_culling = enable;
  }

  /////////////////////////////////////////////////////////////////////////////
  bool                          
  volume_renderer::detect_faces_by_sampling    () const
  {
    return _detect_face_by_sampling;
  }

  /////////////////////////////////////////////////////////////////////////////
  void                  
  volume_renderer::detect_faces_by_sampling    ( bool enable )
  {
    _detect_face_by_sampling = enable;
  }

  /////////////////////////////////////////////////////////////////////////////
  bool                          
  volume_renderer::detect_implicit_inflection    () const
  {
    return _detect_implicit_inflection;
  }

  /////////////////////////////////////////////////////////////////////////////
  void                  
  volume_renderer::detect_implicit_inflection    ( bool enable )
  {
    _detect_implicit_inflection = enable;
  }

  /////////////////////////////////////////////////////////////////////////////
  bool                          
  volume_renderer::detect_implicit_extremum      () const
  {
    return _detect_implicit_extremum;
  }

  /////////////////////////////////////////////////////////////////////////////
  void                  
  volume_renderer::detect_implicit_extremum      ( bool enable )
  {
    _detect_implicit_extremum = enable;
  }


  /////////////////////////////////////////////////////////////////////////////
  visualization_properties const&                           
  volume_renderer::visualization_props () const
  {
    return _visualization_props;
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */  void                  
  volume_renderer::visualization_props ( visualization_properties const& props )
  {
    _visualization_props = props;
  }


  /////////////////////////////////////////////////////////////////////////////
  void volume_renderer::set_external_texture ( std::shared_ptr<gpucast::gl::texture2d> const& color_depth_texture )
  {
    _external_color_depth_texture = color_depth_texture;
  }
  

  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::_init()
  {
    // init transfertexture
    gpucast::gl::transferfunction<gpucast::gl::vec4f> tf;

    tf.set(0,   gpucast::gl::vec4f(0.0, 0.0, 1.0, 1.0));
    tf.set(70,  gpucast::gl::vec4f(0.0, 1.0, 1.0, 1.0));
    tf.set(120, gpucast::gl::vec4f(0.0, 1.0, 0.0, 1.0));
    tf.set(190, gpucast::gl::vec4f(1.0, 1.0, 0.0, 1.0));
    tf.set(255, gpucast::gl::vec4f(1.0, 0.0, 0.0, 1.0));

    std::vector<gpucast::gl::vec4f> samples;
    tf.evaluate(256, samples, gpucast::gl::piecewise_linear());

    _transfertexture.reset     ( new gpucast::gl::texture2d );
    _transfertexture->teximage(0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_FLOAT, &samples[0]);

    _transfertexture->set_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _transfertexture->set_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    _transfertexture->set_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP);
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::_init_shader ()
  {
    if ( !_base_program ) 
    {
      init_program( _base_program, "/base/base.vert", "/base/base.frag" );
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  std::string
  volume_renderer::_open ( std::string const& filename ) const
  {
    std::pair<bool, std::string> path_to_file = _path_to_file(filename);

    if (path_to_file.first)
    {
      std::ifstream filestream  ( path_to_file.second.c_str() );
      std::string   source_code ( std::istreambuf_iterator<char>(filestream), (std::istreambuf_iterator<char>()));
      return source_code;
    } else {
      throw std::runtime_error("volume_renderer::_open(): Couldn't open file.");
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer::_init_cuda()
  {
    if (!_initialized_cuda)
    {
      std::cout << "cudaGLSetGLDevice ... " << std::endl;
      cudaError_t cuda_err = cudaGLSetGLDevice(_cuda_get_max_flops_device_id());
      _initialized_cuda = true;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void                              
  volume_renderer::_verify_convexhull ( std::vector<gpucast::math::point3d>&   vertices,
                                        std::vector<int>&            indices ) const
  {
    // determine center
    gpucast::math::point3d center;
    std::for_each(vertices.begin(), vertices.end(), [&] ( gpucast::math::point3f const& p ) { center += p; } );
    center /= gpucast::math::point3f::value_type(vertices.size());

    int baseid = 0;
    while ( baseid <= int(indices.size() - 3) )
    {
      gpucast::math::point3d normal = gpucast::math::cross ( vertices[indices[baseid + 1]] - vertices[indices[baseid    ] ],
                                         vertices[indices[baseid + 2]] - vertices[indices[baseid + 1] ]);
      normal.normalize();

      gpucast::math::point3d delta  = vertices[indices[baseid]] - center;
      delta.normalize();

      if ( dot ( normal, delta ) < 0.0 )
      {
        int tmp = indices[baseid];
        indices[baseid] = indices[baseid + 2];
        indices[baseid + 2] = tmp;
      }

      baseid += 3; // go to next triangle
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  int
  volume_renderer::_cuda_get_max_flops_device_id() const
  {
      int current_device = 0, sm_per_multiproc = 0;
      int max_compute_perf = 0, max_perf_device = 0;
      int device_count = 0, best_SM_arch = 0;
      cudaDeviceProp deviceProp;

      cudaGetDeviceCount(&device_count);
      // Find the best major SM Architecture GPU device
      while (current_device < device_count) {
        cudaGetDeviceProperties(&deviceProp, current_device);
        if (deviceProp.major > 0 && deviceProp.major < 9999) {
          best_SM_arch = std::max(best_SM_arch, deviceProp.major);
        }
        current_device++;
      }

      // Find the best CUDA capable GPU device
      current_device = 0;
      while (current_device < device_count) {
        cudaGetDeviceProperties(&deviceProp, current_device);
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
          sm_per_multiproc = 1;
        }
        else {
          sm_per_multiproc = _ConvertSMVer2Cores_local(deviceProp.major, deviceProp.minor);
        }

        int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        if (compute_perf  > max_compute_perf) {
          // If we find GPU with SM major > 2, search only these
          if (best_SM_arch > 2) {
            // If our device==dest_SM_arch, choose this, or else pass
            if (deviceProp.major == best_SM_arch) {
              max_compute_perf = compute_perf;
              max_perf_device = current_device;
            }
          }
          else {
            max_compute_perf = compute_perf;
            max_perf_device = current_device;
          }
        }
        ++current_device;
      }
      return max_perf_device;
    }


} // namespace gpucast
