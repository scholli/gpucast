/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : fragmentlist_generator.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_fragmentlist_generator_HPP
#define GPUCAST_fragmentlist_generator_HPP

// header, system

// header, external
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/gl/primitives/plane.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/texturebuffer.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/volume_renderer.hpp>
#include <gpucast/volume/beziervolume.hpp>

#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp>
#include <gpucast/math/oriented_boundingbox_random_policy.hpp>
#include <gpucast/math/oriented_boundingbox_axis_aligned_policy.hpp>
#include <gpucast/math/oriented_boundingbox_covariance_policy.hpp>
#include <gpucast/math/oriented_boundingbox_greedy_policy.hpp>

namespace gpucast 
{

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME fragmentlist_generator : public volume_renderer
{
public : // enums, typedefs

  /////////////////////////////////////////////////////////////////////////////
  // per drawble ressource
  /////////////////////////////////////////////////////////////////////////////
  struct drawable_ressource_impl
  {
    drawable_ressource_impl()
      : size        ( 0 ),
        initialized ( false )
    {}
  
    unsigned                    size;

    bool                        initialized;

    gpucast::gl::arraybuffer           attribarray0;
    gpucast::gl::arraybuffer           attribarray1;
  
    gpucast::gl::elementarraybuffer    indexarray;
  
    gpucast::gl::vertexarrayobject     vao;
  };

  typedef std::shared_ptr<gpucast::gl::texturebuffer>                        texturebuffer_ptr;
  typedef std::shared_ptr<drawable_ressource_impl>                    drawable_ressource_ptr;
  typedef boost::unordered_map<drawable_ptr, drawable_ressource_ptr>    drawable_map;
  typedef drawable_map::value_type                                      drawable_ressource_pair;
  typedef beziervolume::point_type                                      point_type;

  enum proxy_type { convex_hull = 0, 
                    parallipiped = 1, 
                    count = 2
                  };

public : // c'tor / d'tor

  fragmentlist_generator            ( int argc, char** argv );
  virtual ~fragmentlist_generator   ();

public : // methods

  virtual void                            clear                   ();

  bool                                    initialized             () const;     

  virtual void                            draw                    ();

  virtual void                            transform               ( gpucast::math::matrix4f const& m );

  virtual void                            compute_nearfar         ();

  virtual void                            recompile               ();

  virtual void                            resize                  ( int width, int height );

  void                                    readback_information    ( bool enable );

  void                                    allocation_grid_width   ( unsigned width );
  unsigned                                allocation_grid_width   () const;

  void                                    allocation_grid_height  ( unsigned width );
  unsigned                                allocation_grid_height  () const;

  void                                    pagesize                ( unsigned s ); 
  unsigned                                pagesize                () const; 

  void                                    readback                ( );

  unsigned                                usage_fragmentlist      ( ) const;
  unsigned                                usage_fragmentdata      ( ) const;

  virtual void                            write                   ( std::ostream& os ) const;
  virtual void                            read                    ( std::istream& is );

protected : // auxilliary methods

  void                                    _initialize_gl_resources();

  void                                    _initialize_vao         ();

  void                                    _upload_indexarrays     ();
  void                                    _upload_vertexarrays    ();
  void                                    _initialize_texbuffers  ();

  void                                    _init                   ();
  void                                    _init_shader            ();

private :  // auxilliary methods

  void                                    _clear_images           ();
  void                                    _generate_fragmentlists ();
  void                                    _sort_fragmentlists     ();
  
protected : // attributes
                        
  std::vector<gpucast::math::vec4u>      _surface_data;  
  //-----------------------------------------------------------------------------   
  // [unique_surface_id] [volume_outer_cell]           [inner/outer]  [uid]         
  // [unique_volume_id]  [surface_mesh_base_id]        [order_u]      [vid]         
  // [volume_data_id]    [adjacent_volume_data_id]     [order_v]      [wid]         
  // [attribute_data_id] [adjacent_attribute_data_id]  [surface_type] [fixed_param] 
  //-----------------------------------------------------------------------------

  std::vector<gpucast::math::vec4f>      _surface_points;
  //-----------------------------------------------------------------------------
  // [wx_00] [wx_01] ...
  // [wy_00] [wy_01] ...
  // [wz_00] [wz_01] ...
  // [ w_00] [ w_01] ...
  //-----------------------------------------------------------------------------

  std::vector<gpucast::math::vec4f>      _volume_data; 
  //-----------------------------------------------------------------------------
  // [ volume_points_id  ] [ order_u ]        [ umin_local ] [ umax_local ] [ umin_global ] [ umax_global ] [ surface_id_umin ] [ surface_id_wmax ]
  // [ uid               ] [ order_v ]        [ vmin_local ] [ vmax_local ] [ vmin_global ] [ vmax_global ] [ surface_id_umax ] [ surface_id_wmax ] 
  // [ attribute_data_id ] [ order_w ]        [ wmin_local ] [ wmax_local ] [ wmin_global ] [ wmax_global ] [ surface_id_vmin ] [ 0 ]
  // [ bbox_diameter     ] [ has_outer_face ] [ 0 ]          [ 0 ]          [ 0 ]           [ 0 ]           [ surface_id_vmax ] [ 0 ]
  //-----------------------------------------------------------------------------

  std::vector<gpucast::math::vec4f>      _volume_points;
  //-----------------------------------------------------------------------------
  // volume_points_id
  // [ wx_000 ]      [ wx_001 ] ...
  // [ wy_000 ]      [ wy_001 ] ...
  // [ wz_000 ]      [ wz_001 ] ...
  // [  w_000 ]      [  w_001 ] ...
  //-----------------------------------------------------------------------------

  std::vector<gpucast::math::vec4f>      _attribute_data; // attribute data buffers
  // attribute_data_id
  //-----------------------------------------------------------------------------
  // [ min ]       
  // [ max ]       
  // [ attribute_point_id ]  
  // [ empty ]
  //-----------------------------------------------------------------------------

  std::vector<gpucast::math::vec2f>      _attribute_points; // attribute buffers
  // attribute_point_id
  //-----------------------------------------------------------------------------
  // [attrib0_000] ...
  // [attrib1_000] ...
  //-----------------------------------------------------------------------------

  drawable_ressource_ptr                          _drawable;

  // client side buffer for vertex array object
  bool                                            _object_initialized;
  bool                                            _gl_initialized;

  gpucast::math::matrix4f                                  _modelmatrix;

  std::vector<gpucast::math::vec3f>                        _vertices;          // vertices of convex hull
  std::vector<gpucast::math::vec4f>                        _vertexparameter;   // [u, v, 0, surface_id]
  renderinfo                                      _renderinfo;        // binning for attribute

  //-----------------------------------------------------------------------------
  gpucast::gl::texturebuffer                             _volume_data_texturebuffer;
  gpucast::gl::texturebuffer                             _volume_points_texturebuffer;
  gpucast::gl::texturebuffer                             _surface_data_texturebuffer;
  gpucast::gl::texturebuffer                             _surface_points_texturebuffer;
  gpucast::gl::texturebuffer                             _attribute_data_texturebuffer;
  gpucast::gl::texturebuffer                             _attribute_points_texturebuffer;

  std::shared_ptr<gpucast::gl::texture2d>              _indextexture;
  std::shared_ptr<gpucast::gl::texture2d>              _semaphoretexture;
  std::shared_ptr<gpucast::gl::texture2d>              _fragmentcount;

  std::shared_ptr<gpucast::gl::texturebuffer>          _allocation_grid;
  std::shared_ptr<gpucast::gl::texturebuffer>          _indexlist;
  std::shared_ptr<gpucast::gl::texturebuffer>          _master_counter;

  // parameters 
  bool                                            _discard_by_minmax;
  bool                                            _enable_sorting;
  bool                                            _render_side_chulls;

  bool                                            _readback;
  unsigned                                        _usage_indexbuffer;
  unsigned                                        _usage_fragmentbuffer;

  unsigned                                        _allocation_grid_width;
  unsigned                                        _allocation_grid_height;
  unsigned                                        _pagesize;
  unsigned                                        _pagesize_per_fragment;
  unsigned                                        _maxsize_fragmentbuffer; 
  unsigned                                        _maxsize_fragmentdata;

  std::shared_ptr<gpucast::gl::program>                _clean_pass;
  std::shared_ptr<gpucast::gl::program>                _hull_pass;
  std::shared_ptr<gpucast::gl::program>                _sort_pass;

  std::shared_ptr<gpucast::gl::plane>                  _quad;
};

} // namespace gpucast

#endif // GPUCAST_fragmentlist_generator_HPP
