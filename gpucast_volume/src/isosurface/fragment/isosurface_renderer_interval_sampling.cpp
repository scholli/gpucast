/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface_renderer_interval_sampling.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/fragment/isosurface_renderer_interval_sampling.hpp"

// header, system
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

// header, project
#include <gpucast/core/util.hpp>
#include <gpucast/core/hyperspace_adapter.hpp>
#include <gpucast/core/convex_hull_impl.hpp>
#include <gpucast/core/uvgenerator.hpp>
#include <gpucast/volume/uvwgenerator.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>

#include <gpucast/gl/primitives/cube.hpp>

///////////////////////////////////////////////////////////////////////////////
extern "C" void invoke_face_interval_raycasting ( unsigned                     width, 
                                                  unsigned                     height,
                                                  int2                         tilesize,
                                                  unsigned                     pagesize,
                                                  unsigned                     backface_culling,
                                                  float                        nearplane,
                                                  float                        farplane,
                                                  float3                       background,
                                                  float                        iso_threshold,
                                                  int                          adaptive_sampling,
                                                  float                        min_sample_distance,
                                                  float                        max_sample_distance,
                                                  float                        adaptive_sample_scale,
                                                  int                          screenspace_newton_error,
                                                  float                        fixed_newton_epsilon,
                                                  unsigned                     max_iterations_newton,
                                                  unsigned                     max_steps_binary_search,
                                                  float                        global_attribute_min,
                                                  float                        global_attribute_max,
                                                  float                        surface_transparency,
                                                  float                        isosurface_transparency,
                                                  bool                         show_samples_isosurface_intersection,
                                                  bool                         show_samples_face_intersection,
                                                  bool                         show_face_intersections,
                                                  bool                         show_face_intersection_tests,
                                                  bool                         show_isosides,
                                                  bool                         detect_face_by_sampling,
                                                  bool                         detect_implicit_extremum,
                                                  bool                         detect_implicit_inflection,
                                                  struct cudaGraphicsResource* matrices_resource,
                                                  struct cudaGraphicsResource* colorbuffer_resource,
                                                  struct cudaGraphicsResource* depthbuffer_resource,
                                                  struct cudaGraphicsResource* headpointer_resource,
                                                  struct cudaGraphicsResource* fragmentcount_resource,
                                                  struct cudaGraphicsResource* indexlist_resource,
                                                  struct cudaGraphicsResource* allocation_grid_resource,
                                                  struct cudaGraphicsResource* surface_data_buffer_resource,
                                                  struct cudaGraphicsResource* surface_points_buffer_resource,
                                                  struct cudaGraphicsResource* volume_data_buffer_resource,
                                                  struct cudaGraphicsResource* volume_points_buffer_resource,
                                                  struct cudaGraphicsResource* attribute_data_buffer_resource,
                                                  struct cudaGraphicsResource* attribute_points_buffer_resource,
                                                  struct cudaGraphicsResource* external_color_depth_texture );

namespace gpucast {

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_interval_sampling::isosurface_renderer_interval_sampling( int argc, char** argv )
    : basetype             ( argc, argv ),
      _proxy_type          ( convex_hull ),
      _bin_split_heuristic ( new greedy_split(100000, 0.3f, 50) )
  {
    _discard_by_minmax   = false;
    _backface_culling    = false;
    _initialized_cuda    = false;
    _render_side_chulls  = true;
  }

  /////////////////////////////////////////////////////////////////////////////
  isosurface_renderer_interval_sampling::~isosurface_renderer_interval_sampling()
  {}

  /////////////////////////////////////////////////////////////////////////////
  void 
  isosurface_renderer_interval_sampling::raycast_fragment_lists ()
  {
    int2 tilesize = {_allocation_grid_width, _allocation_grid_height};

    // compute relative threshold: threshold relative value between 0 and 1
    float attribute_minimum     = _global_attribute_bounds.minimum();
    float attribute_maximum     = _global_attribute_bounds.maximum();

    float attribute_range       = _global_attribute_bounds.length();
    float normalized_threshold  = attribute_minimum + attribute_range * _relative_isovalue;

    float3 background     = { _background[0], _background[1], _background[2] };

    if ( !_drawable )
    {
      invoke_external_passthrough ( _width, _height, _cuda_colorbuffer, _cuda_depthbuffer, _cuda_external_texture );
    } else {
      // draw all objects
      invoke_face_interval_raycasting ( _width, 
                                        _height, 
                                        tilesize,
                                        _pagesize,
                                        _backface_culling,
                                        _nearplane,
                                        _farplane,
                                        background,
                                        normalized_threshold,
                                        _adaptive_sampling,
                                        _min_sample_distance,
                                        _max_sample_distance,
                                        _adaptive_sample_scale,
                                        _screenspace_newton_error,
                                        _newton_epsilon,
                                        _newton_iterations,
                                        _max_binary_searches,
                                        attribute_minimum,
                                        attribute_maximum,
                                        _surface_transparency,
                                        _isosurface_transparency,
                                        _visualization_props.show_samples_isosurface_intersection,
                                        _visualization_props.show_samples_face_intersection,
                                        _visualization_props.show_face_intersections,
                                        _visualization_props.show_face_intersection_tests,
                                        _visualization_props.show_isosides,
                                        _detect_face_by_sampling,
                                        _detect_implicit_extremum,
                                        _detect_implicit_inflection,
                                        _cuda_matrixbuffer,
                                        _cuda_colorbuffer, 
                                        _cuda_depthbuffer, 
                                        _cuda_headpointer, 
                                        _cuda_fragmentcount,
                                        _cuda_indexlist, 
                                        _cuda_allocation_grid, 
                                        _cuda_surface_data_buffer,
                                        _cuda_surface_points_buffer,
                                        _cuda_volume_data_buffer,
                                        _cuda_volume_points_buffer,
                                        _cuda_attribute_data_buffer,
                                        _cuda_attribute_points_buffer,
                                        _cuda_external_texture
                                      );
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  bool 
  isosurface_renderer_interval_sampling::initialize ( std::string const& attribute_name )
  {
    if ( _surface_data.empty()     ) _surface_data.resize(1); 
    if ( _surface_points.empty()   ) _surface_points.resize(1);
    if ( _volume_data.empty()      ) _volume_data.resize(1);
    if ( _volume_points.empty()    ) _volume_points.resize(1); 
    if ( _attribute_data.empty()   ) _attribute_data.resize(1); 
    if ( _attribute_points.empty() ) _attribute_points.resize(1); 
  
    // clear vertex arrays
    _vertices.clear();
    _vertexparameter.clear();

    // initialize helper structures
    std::vector<deferred_surface_header_write>  deferred_surface_jobs;

    std::map<unsigned, unsigned>                uniqueid_volume_index_map; 
    std::map<unsigned, unsigned>                uniqueid_attribute_buffer_index_map; 

    // initialize with one renderbin
    _renderinfo.clear();
    _renderinfo = renderinfo (  get_attributebounds() );
  
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

      gpucast::math::vec4u surface_info0 ( job.surface_uid,
                                  job.volume_uid,
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

      // create renderchunk - get minimum and maximum of surface attached volume
      renderinfo::value_type amin = _attribute_data[attribute_data_id][0];
      renderinfo::value_type amax = _attribute_data[attribute_data_id][1];

      // merge with adjacent volume if there
      if ( adjacent_attribute_data_id != 0 )
      {
        amin = std::min ( amin, _attribute_data[adjacent_attribute_data_id][0]);
        amax = std::max ( amax, _attribute_data[adjacent_attribute_data_id][1]);
      }

      renderchunk_ptr chunk ( new renderchunk ( job.indices, renderchunk::interval_type( amin, amax, gpucast::math::included, gpucast::math::included), job.outer_face ));
      _renderinfo.insert(chunk);
    } 

    std::cout << std::endl;

    // optimize renderinfo
    _renderinfo.optimize  ( *_bin_split_heuristic );
    _renderinfo.serialize ( );

    std::cout << "Finished binning for Beziervolumeobject : " <<std::endl;
    std::cout << "Renderinfo for attribute : " << attribute_name << " results in " << _renderinfo.renderbins_size () << std::endl;
    for ( auto k = _renderinfo.renderbin_begin(); k != _renderinfo.renderbin_end(); ++k )
    {
      std::cout << "bin : " << k->chunks() << " face proxies" << std::endl;
    }
  
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_interval_sampling::_create_proxy_geometry ( beziervolume const& v, 
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

        // deferred fill job, because not all neighbor volumes are stored yet -> not all buffer id's are known
        deferred_surface_header_write   job;

        switch ( _proxy_type )
        {
          case convex_hull  : 
            _build_convex_hull (v, beziervolume::boundary_t(surface), job ); 
            break;
          case paralleliped : 
            _build_paralleliped (v, beziervolume::boundary_t(surface), job ); 
            break;
          default : 
            throw std::runtime_error("Proxy type not implemented");
        };

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

  ////////////////////////////////////////////////////////////////////////////////
  void 
  isosurface_renderer_interval_sampling::_build_convex_hull ( beziervolume const&            v, 
                                                             enum beziervolume::boundary_t   face_type,
                                                             deferred_surface_header_write&  job )
  {
    // get base index
    unsigned vertex_base_index = unsigned ( _vertices.size() );
    unsigned surface_data_id   = unsigned ( _surface_data.size() );

    // compute 6 outer surfaces serving as boundary to the volume
    gpucast::math::beziersurface<gpucast::math::vec3d> outer_face;
    switch ( face_type )
    {
      case beziervolume::boundary_t::umin : outer_face = v.slice<gpucast::math::vec3d> (point_type::x, 0); break;
      case beziervolume::boundary_t::umax : outer_face = v.slice<gpucast::math::vec3d> (point_type::x, v.degree_u() ); break;
      case beziervolume::boundary_t::vmin : outer_face = v.slice<gpucast::math::vec3d> (point_type::y, 0 ); break;
      case beziervolume::boundary_t::vmax : outer_face = v.slice<gpucast::math::vec3d> (point_type::y, v.degree_v() ); break;
      case beziervolume::boundary_t::wmin : outer_face = v.slice<gpucast::math::vec3d> (point_type::z, 0 ); break;
      case beziervolume::boundary_t::wmax : outer_face = v.slice<gpucast::math::vec3d> (point_type::z, v.degree_w() ); break;
    }

    job.order_u                  = unsigned(outer_face.order_u());
    job.order_v                  = unsigned(outer_face.order_v());

    // generate local convex hull of outer surface
    std::vector<gpucast::math::point3d> surface_chull_vertices;
    surface_chull_vertices.reserve ( outer_face.order_u() * outer_face.order_v() * 3);

    std::vector<int>          surface_chull_indices;
    surface_chull_indices.reserve  ( outer_face.order_u() * outer_face.order_v() * 3);

    convex_hull_compute<3, gpucast::math::point3d>(&(*(outer_face.begin()))[0],
                                        outer_face.size(),
                                        std::back_inserter(surface_chull_vertices),
                                        std::back_inserter(surface_chull_indices),
                                        0);

    // verify that all triangles are in CCW order
    _verify_convexhull ( surface_chull_vertices, surface_chull_indices );

    // copy convex hull of side surface to global buffer and apply necessary buffer offsets
    std::size_t vertices_offset = _vertices.size();
    _vertices.resize ( vertices_offset + surface_chull_vertices.size() );
    std::copy      ( surface_chull_vertices.begin(), surface_chull_vertices.end(), _vertices.begin() + vertices_offset);

    job.indices.resize ( surface_chull_indices.size() );
    std::transform ( surface_chull_indices.begin(),  
                     surface_chull_indices.end(), 
                     job.indices.begin(), [&] ( int i ) 
                     { 
                       return i + vertex_base_index; 
                     } );

    // attach according uvw-parameter to vertex
    std::size_t parameter_osize = _vertexparameter.size();
    _vertexparameter.resize(parameter_osize + outer_face.size());
      
    switch ( face_type ) 
    {
      case beziervolume::boundary_t::umin : std::generate(_vertexparameter.begin() + parameter_osize, _vertexparameter.end(), uvgenerator<gpucast::math::vec4f, 0, 1, 2, 3>(outer_face.order_u(), outer_face.order_v(), 0, 1, 0, 1,-1.0, bit_cast<unsigned,float>(surface_data_id))); break;
      case beziervolume::boundary_t::umax : std::generate(_vertexparameter.begin() + parameter_osize, _vertexparameter.end(), uvgenerator<gpucast::math::vec4f, 0, 1, 2, 3>(outer_face.order_u(), outer_face.order_v(), 0, 1, 0, 1, 2.0, bit_cast<unsigned,float>(surface_data_id))); break;
      case beziervolume::boundary_t::vmin : std::generate(_vertexparameter.begin() + parameter_osize, _vertexparameter.end(), uvgenerator<gpucast::math::vec4f, 0, 1, 2, 3>(outer_face.order_u(), outer_face.order_v(), 0, 1, 0, 1,-1.0, bit_cast<unsigned,float>(surface_data_id))); break;
      case beziervolume::boundary_t::vmax : std::generate(_vertexparameter.begin() + parameter_osize, _vertexparameter.end(), uvgenerator<gpucast::math::vec4f, 0, 1, 2, 3>(outer_face.order_u(), outer_face.order_v(), 0, 1, 0, 1, 2.0, bit_cast<unsigned,float>(surface_data_id))); break;
      case beziervolume::boundary_t::wmin : std::generate(_vertexparameter.begin() + parameter_osize, _vertexparameter.end(), uvgenerator<gpucast::math::vec4f, 0, 1, 2, 3>(outer_face.order_u(), outer_face.order_v(), 0, 1, 0, 1,-1.0, bit_cast<unsigned,float>(surface_data_id))); break;
      case beziervolume::boundary_t::wmax : std::generate(_vertexparameter.begin() + parameter_osize, _vertexparameter.end(), uvgenerator<gpucast::math::vec4f, 0, 1, 2, 3>(outer_face.order_u(), outer_face.order_v(), 0, 1, 0, 1, 2.0, bit_cast<unsigned,float>(surface_data_id))); break;
    };

    // pushback placeholder for surface header data
    _surface_data.resize( _surface_data.size() + volume_renderer::surface_data_header );

    // transform control points of outer surface and copy to surface-buffer
    std::transform (outer_face.begin(), outer_face.end(), std::back_inserter(_surface_points), hyperspace_adapter_3D_to_4D<gpucast::math::point3d, gpucast::math::vec4f>());
  }


  ////////////////////////////////////////////////////////////////////////////////
  void isosurface_renderer_interval_sampling::_build_paralleliped ( beziervolume const&             v, 
                                                                   enum beziervolume::boundary_t   face_type,
                                                                   deferred_surface_header_write&  job )
  {
    // get base index
    unsigned vertex_base_index = unsigned ( _vertices.size() );
    unsigned param_base_index  = unsigned ( _vertexparameter.size() );
    unsigned surface_data_id  = unsigned ( _surface_data.size() );

    // compute 6 outer surfaces serving as boundary to the volume
    gpucast::math::beziersurface<gpucast::math::point3d> outer_face;
    switch ( face_type )
    {
      case beziervolume::boundary_t::umin : outer_face = v.slice<gpucast::math::point3d> (point_type::x, 0); break;
      case beziervolume::boundary_t::umax : outer_face = v.slice<gpucast::math::point3d> (point_type::x, v.degree_u() ); break;
      case beziervolume::boundary_t::vmin : outer_face = v.slice<gpucast::math::point3d> (point_type::y, 0 ); break;
      case beziervolume::boundary_t::vmax : outer_face = v.slice<gpucast::math::point3d> (point_type::y, v.degree_v() ); break;
      case beziervolume::boundary_t::wmin : outer_face = v.slice<gpucast::math::point3d> (point_type::z, 0 ); break;
      case beziervolume::boundary_t::wmax : outer_face = v.slice<gpucast::math::point3d> (point_type::z, v.degree_w() ); break;
    }

    job.order_u                  = unsigned(outer_face.order_u());
    job.order_v                  = unsigned(outer_face.order_v());

    // create triangle mesh for bounding box and copy to vertex array
    gpucast::math::obbox3d obb ( outer_face.mesh(), gpucast::math::partial_derivative_policy<gpucast::math::obbox3d::point_type>());
    std::vector<gpucast::math::obbox3d::point_type> corner_vertices;
    obb.generate_corners ( std::inserter ( corner_vertices, corner_vertices.begin() ) );
  
    std::vector<int>         paralleliped_indices;
    for (unsigned i = 0; i < 6 * 2 * 3; ++i) { // 6 sides and two triangles with 3 vertices 
      paralleliped_indices.push_back(i);
    }

    std::vector<gpucast::math::vec4f> paralleliped_vertices = gpucast::gl::cube::create_triangle_mesh(corner_vertices[0], corner_vertices[1],
                                                                                                    corner_vertices[2], corner_vertices[3],
                                                                                                    corner_vertices[4], corner_vertices[5],
                                                                                                    corner_vertices[6], corner_vertices[7]);

    _vertices.resize ( vertex_base_index + paralleliped_vertices.size() );
    std::copy ( paralleliped_vertices.begin(), paralleliped_vertices.end(), _vertices.begin() + vertex_base_index);

    // create attribute array [u v type surface_id]
    float attrib0 = 0.0f;
    float attrib1 = bit_cast<unsigned, float>(surface_data_id);

    switch ( face_type ) 
    {
      case beziervolume::boundary_t::umin : attrib0 = -1.0f; break;
      case beziervolume::boundary_t::umax : attrib0 =  2.0f; break;
      case beziervolume::boundary_t::vmin : attrib0 = -1.0f; break;
      case beziervolume::boundary_t::vmax : attrib0 =  2.0f; break;
      case beziervolume::boundary_t::wmin : attrib0 = -1.0f; break;
      case beziervolume::boundary_t::wmax : attrib0 =  2.0f; break;
    };

    std::vector<gpucast::math::vec4f> paralleliped_parameter =
      gpucast::gl::cube::create_triangle_mesh(
        gpucast::math::vec4f(0.0f, 0.0f, attrib0, attrib1),
        gpucast::math::vec4f(1.0f, 0.0f, attrib0, attrib1),
        gpucast::math::vec4f(0.0f, 1.0f, attrib0, attrib1),
        gpucast::math::vec4f(1.0f, 1.0f, attrib0, attrib1),
        gpucast::math::vec4f(0.0f, 0.0f, attrib0, attrib1),
        gpucast::math::vec4f(1.0f, 0.0f, attrib0, attrib1),
        gpucast::math::vec4f(0.0f, 1.0f, attrib0, attrib1),
        gpucast::math::vec4f(1.0f, 1.0f, attrib0, attrib1)
      );
                                       

    // copy parameter to attribute array
    _vertexparameter.resize ( param_base_index + paralleliped_parameter.size() );
    std::copy ( paralleliped_parameter.begin(), paralleliped_parameter.end(), _vertexparameter.begin() + param_base_index);

    // set vertex indices
    job.indices.resize ( 36 );
    std::transform ( paralleliped_indices.begin(),
                     paralleliped_indices.end(),
                     job.indices.begin(), 
                     [&] ( int i ) { return vertex_base_index + i; } );

    // pushback placeholder for surface header data
    _surface_data.resize( _surface_data.size() + volume_renderer::surface_data_header );

    // transform control points of outer surface and copy to surface-buffer
    std::transform (outer_face.begin(), outer_face.end(), std::back_inserter(_surface_data), hyperspace_adapter_3D_to_4D<gpucast::math::point3d, gpucast::math::vec4f>());
  }


} // namespace gpucast
