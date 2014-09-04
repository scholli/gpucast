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
#ifndef GPUCAST_VOLUME_RENDERER_HPP
#define GPUCAST_VOLUME_RENDERER_HPP

// header, system
#include <set>
#include <map>
#include <string>

// header, external
#include <memory>
#include <boost/unordered_map.hpp>
#include <boost/noncopyable.hpp>

#include <gpucast/math/vec4.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/primitives/coordinate_system.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/core/renderer.hpp>
#include <gpucast/volume/beziervolumeobject.hpp>

extern "C" void invoke_external_passthrough ( unsigned _width,
                                              unsigned _height,
                                              struct cudaGraphicsResource* colorbuffer_resource,
                                              struct cudaGraphicsResource* depthbuffer_resource, 
                                              struct cudaGraphicsResource* cuda_external_texture );

namespace gpucast {

class beziervolumeobject;

////////////////////////////////////////////////////////////////////////////////
struct visualization_properties 
{
  visualization_properties() 
    : show_samples_isosurface_intersection(false),
      show_samples_face_intersection(false),
      show_face_intersections(false),
      show_face_intersection_tests(false),
      show_isosides(false)
  {}

  bool show_samples_isosurface_intersection;
  bool show_samples_face_intersection;
  bool show_face_intersections;
  bool show_face_intersection_tests;
  bool show_isosides;
};

////////////////////////////////////////////////////////////////////////////////
// helper structure to write volume indices to buffer of volumes that are not in the buffer
////////////////////////////////////////////////////////////////////////////////
struct deferred_surface_header_write
{
  bool              outer_face;
  bool              outer_cell;
  std::vector<int>  indices;

  unsigned          order_u;
  unsigned          order_v;

  unsigned          surface_uid;
  unsigned          surface_points_id;
  unsigned          surface_data_id;
  unsigned          surface_type;
    
  unsigned          volume_uid;
  unsigned          adjacent_volume_uid;
};

class GPUCAST_VOLUME volume_renderer : public renderer
{
public : // enums, typedefs

  static unsigned const                                                                volume_data_header  = 8;
  static unsigned const                                                                surface_data_header = 4;

  typedef beziervolumeobject                                                           drawable_type;
  typedef std::shared_ptr<drawable_type>                                             drawable_ptr;
  typedef std::shared_ptr<gpucast::gl::texture2d>                                           texture2d_ptr;
  typedef gpucast::math::interval<beziervolume::attribute_type::value_type>                      attribute_interval;

  enum proxy_type { convex_hull  = 0, // convex hull 
                    paralleliped = 1, // object-oriented bounding box
                    count        = 2 };

public : // c'tor / d'tor

  volume_renderer           ( int argc, char** argv );
  virtual ~volume_renderer  ();

public : // pure virtual methods
                 
  virtual void                  init                    ( drawable_ptr const&     object,
                                                          std::string const&      attribute_name ) = 0;
                                                          //enum                    proxy_type = convex_hull,
                                                          //split_heuristic const&  split_heuristic = greedy_split(10000, 0.3f, 50)) = 0;

  virtual void                  set                      ( drawable_ptr const&, std::string const& attribute_name );

  virtual void                  clear                    ();  
                                                         
  virtual void                  recompile                ();
                                                         
  virtual void                  draw                     () = 0;
                                                         
  virtual void                  transform                ( gpucast::math::matrix4f const& m ) = 0;
                                                         
public : // methods                                      
                                                                                                               
  attribute_interval const&     get_attributebounds      ( ) const;
  virtual void                  set_attributebounds      ( attribute_interval const& m );                                     

  unsigned                      newton_iterations        ( ) const;
  virtual void                  newton_iterations        ( unsigned );
                                                         
  float                         newton_epsilon           ( ) const;
  virtual void                  newton_epsilon           ( float );
                                                         
  float                         relative_isovalue        () const;
  virtual void                  relative_isovalue        ( float ); 
                                                         
  bool                          adaptive_sampling        () const;
  virtual void                  adaptive_sampling        ( bool );

  bool                          screenspace_newton_error () const;
  virtual void                  screenspace_newton_error ( bool );
                   
  float                         min_sample_distance      () const;
  virtual void                  min_sample_distance      ( float );
                                                         
  float                         max_sample_distance      () const;
  virtual void                  max_sample_distance      ( float );
                                                         
  float                         adaptive_sample_scale    () const;
  virtual void                  adaptive_sample_scale    ( float );
                                                         
  unsigned                      max_octree_depth         () const;
  virtual void                  max_octree_depth         ( unsigned );
                                                         
  unsigned                      max_volumes_per_node     () const;
  virtual void                  max_volumes_per_node     ( unsigned );
                                                         
  unsigned                      max_binary_searches      () const;
  virtual void                  max_binary_searches      ( unsigned );
                                                         
  float                         isosurface_opacity       () const;
  virtual void                  isosurface_opacity       ( float alpha );

  float                         surface_opacity          () const;
  virtual void                  surface_opacity          ( float alpha );
                                                         
  bool                          backface_culling         () const;
  virtual void                  backface_culling         ( bool enable );

  bool                          detect_faces_by_sampling () const;
  virtual void                  detect_faces_by_sampling ( bool enable );
  
  bool                          detect_implicit_inflection () const;
  virtual void                  detect_implicit_inflection ( bool enable );

  bool                          detect_implicit_extremum   () const;
  virtual void                  detect_implicit_extremum   ( bool enable );

  void                          init_program             ( std::shared_ptr<gpucast::gl::program>&  p,
                                                           std::string const& vertexshader_filename,
                                                           std::string const& fragmentshader_filename,
                                                           std::string const& geometryshader_filename = "");
                                                         
  visualization_properties const& visualization_props    () const;
  virtual void                    visualization_props    ( visualization_properties const& props );

  virtual void                  set_external_texture     ( std::shared_ptr<gpucast::gl::texture2d> const& texture );

  virtual void                  write                    ( std::ostream& os ) const = 0;
  virtual void                  read                     ( std::istream& is ) = 0;

protected :
  
  std::string                   _open                 ( std::string const& filename ) const;
  
  void                          _init                 ();

  void                          _init_cuda            ();

  virtual void                  _init_shader          ();

  void                          _verify_convexhull    ( std::vector<gpucast::math::point3d>& vertices, std::vector<int>& indices ) const;

  int                           _cuda_get_max_flops_device_id() const;
protected : // attributes           

  drawable_ptr                      _object;
  float                             _relative_isovalue;

  bool                              _screenspace_newton_error;
  unsigned                          _newton_iterations;
  float                             _newton_epsilon;
  bool                              _adaptive_sampling;
  bool                              _show_gradients;
  float                             _min_sample_distance;
  float                             _max_sample_distance;
  float                             _adaptive_sample_scale;
  
  unsigned                          _max_octree_depth;
  unsigned                          _max_volumes_per_node;
  unsigned                          _max_binary_searches;

  float                             _surface_transparency;
  float                             _isosurface_transparency;

  bool                              _backface_culling;
              
  bool                              _detect_face_by_sampling;
  bool                              _detect_implicit_inflection;
  bool                              _detect_implicit_extremum;

  visualization_properties          _visualization_props;

  attribute_interval                _global_attribute_bounds;

  // global ressources   
  std::shared_ptr<gpucast::gl::program>   _base_program;
  std::shared_ptr<gpucast::gl::texture2d> _transfertexture;

  std::shared_ptr<gpucast::gl::texture2d> _external_color_depth_texture;
};

template <typename container_t>
inline void write ( std::ostream& os, container_t const& c )
{
  std::size_t out_size = c.size();
  os.write ( reinterpret_cast<char const*>(&out_size), sizeof(std::size_t) );
  os.write ( reinterpret_cast<char const*>(&c.front()), sizeof(container_t::value_type) * out_size );
}

template <typename container_t>
inline void read ( std::istream& is, container_t& c )
{
  std::size_t in_size;
  is.read ( reinterpret_cast<char*>(&in_size), sizeof(std::size_t) );
  c.resize(in_size);
  is.read ( reinterpret_cast<char*>(&c.front()), sizeof(container_t::value_type) * in_size );
}

typedef std::shared_ptr<volume_renderer> volume_renderer_ptr;

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_HPP
