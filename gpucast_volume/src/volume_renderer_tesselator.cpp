/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_tesselator.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/volume_renderer_tesselator.hpp"

// header, system
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/transferfunction.hpp>

// header, project
#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>
#include <gpucast/volume/uid.hpp>

namespace gpucast {

/////////////////////////////////////////////////////////////////////////////
// per drawble ressource
/////////////////////////////////////////////////////////////////////////////
struct volume_renderer_tesselator::drawable_ressource_impl
{
  typedef std::map<std::string, gpucast::math::axis_aligned_boundingbox<gpucast::math::point3d>>        minmaxmap_type;

  drawable_ressource_impl()
    : index       ( 0 ),
      count       ( 0 ),
      modelmatrix ( )
  {}

  unsigned                    index;
  unsigned                    count;

  gpucast::gl::matrix4f              modelmatrix;
};
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  volume_renderer_tesselator::volume_renderer_tesselator( int argc, char** argv )
    : volume_renderer           ( argc, argv ),
      _drawables                ( ),
      _tesselation_depth        ( 0 ),
      _backface_culling         ( true ),
      _initialized              ( false ),
      _render_pass              ()
  {
    _init();
  }


  /////////////////////////////////////////////////////////////////////////////
  volume_renderer_tesselator::~volume_renderer_tesselator()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer_tesselator::clear ()
  {
    volume_renderer::clear();
    _drawables.clear();
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  volume_renderer_tesselator::draw ()
  {
    /*
    if ( _drawables.empty() ) return;

    if (!_render_pass) {
      _init_shader();
    }

    // first of all initialize and update all objects
    BOOST_FOREACH ( drawable_ptr const& p, _objects )
    {
      if ( !_drawables.count(p) ) 
      {
        _drawables.insert ( std::make_pair ( p, drawable_ressource_ptr ( new drawable_ressource_impl ) ) );
        _drawables.find(p)->first->init();
      }
    }

    if ( !_initialized ) 
    {
      _initialize_gl_resources();
      _initialized = true;
    }

    if (_backface_culling)
    {
      glDisable(GL_CULL_FACE);
      glEnable (GL_DEPTH_TEST);
    }

    _render_pass->begin();

    _render_pass->set_uniform1f         ("nearplane",         _nearplane);
    _render_pass->set_uniform1f         ("farplane",          _farplane);
    _render_pass->set_texture2d         ( "transfertexture", *_transfertexture, 0 );

    gpucast::gl::vec4f datamin = _global_attribute_bounds[_current_attribute].min;
    gpucast::gl::vec4f datamax = _global_attribute_bounds[_current_attribute].max;
    _render_pass->set_uniform4fv          ("global_attribute_min", 1,  &datamin[0]);
    _render_pass->set_uniform4fv          ("global_attribute_max", 1,  &datamax[0]);

    _vao.bind();
    {
      _vao.attrib_array(_vertices, 0, 3, GL_FLOAT, false, 0, 0);
      _vao.enable_attrib(0);

      _vao.attrib_array(*(_attributes.at(_current_attribute)), 1, 4, GL_FLOAT, false, 0, 0);
      _vao.enable_attrib(1);

      BOOST_FOREACH(drawable_ressource_pair const& p, _drawables)
      {
        gpucast::gl::matrix4f modelview        = _modelviewmatrix * p.second->modelmatrix;
        gpucast::gl::matrix4f modelviewinverse = gpucast::gl::inverse(modelview);
        gpucast::gl::matrix4f normalmatrix     = modelview.normalmatrix();
        gpucast::gl::matrix4f mvp              = _projectionmatrix * modelview;
  
        _render_pass->set_uniform_matrix4fv ("normalmatrix",              1, false, &normalmatrix[0]);
        _render_pass->set_uniform_matrix4fv ("modelviewmatrix",           1, false, &modelview[0]);
        _render_pass->set_uniform_matrix4fv ("modelviewmatrixinverse",    1, false, &modelviewinverse[0]);
        _render_pass->set_uniform_matrix4fv ("modelviewprojectionmatrix", 1, false, &mvp[0]);

        // bind index array and set draw pointer
        _indexarray.bind();
        glDrawElements(GL_TRIANGLES, GLsizei(p.second->count), GL_UNSIGNED_INT, (GLvoid*)(sizeof(int) * p.second->index) );
        _indexarray.unbind();
      }
    }
    _vao.unbind();

    _render_pass->end();
    */
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer_tesselator::draw ( drawable_ptr const& drawable )
  {
    throw std::runtime_error("volume_renderer_raycast_multipass::draw ( drawable_ptr const& drawable ): not implemented yet!");
  }


  /////////////////////////////////////////////////////////////////////////////
  void                      
  volume_renderer_tesselator::transform ( drawable_ptr const& object, gpucast::gl::matrix4f const& m )
  {
    if ( _drawables.count ( object ) ) 
    {
      _drawables.find( object )->second->modelmatrix = m;
    } else {
      std::cerr << "warning: volume_renderer_tesselator::transform() : can't find drawable to transform" << std::endl;
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer_tesselator::recompile ()
  {
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer_tesselator::_initialize_gl_resources ( )
  {
    /*
    std::map<std::string, std::vector<gpucast::gl::vec4f>> per_vertex_attribute_map;
    std::vector<gpucast::gl::vec3f>                        per_vertex_position;
    std::vector<int>                                indices;

    // allocate arraybuffer for attributes
    BOOST_FOREACH ( auto attribute_name, _attributelist )
    {
      _attributes.insert              ( std::make_pair ( attribute_name, std::shared_ptr<gpucast::gl::arraybuffer>(new gpucast::gl::arraybuffer) ) );
      per_vertex_attribute_map.insert ( std::make_pair ( attribute_name, std::vector<gpucast::gl::vec4f>() ) );
    }

    // copy data to attribute arraybuffer
    BOOST_FOREACH ( drawable_map::value_type const& p, _drawables )
    {
      p.second->index = unsigned(indices.size());

      for ( beziervolumeobject::const_iterator i = p.first->begin(); i != p.first->end(); ++i)
      {
        std::array<gpucast::math::beziersurface<gpucast::gl::vec3d>, 6 > outer;

        outer[0] = (*i).slice<gpucast::gl::vec3d> (gpucast::math::point3d::x, 0);
        outer[1] = (*i).slice<gpucast::gl::vec3d> (gpucast::math::point3d::x, (*i).degree_u() );
        outer[2] = (*i).slice<gpucast::gl::vec3d> (gpucast::math::point3d::y, 0 );
        outer[3] = (*i).slice<gpucast::gl::vec3d> (gpucast::math::point3d::y, (*i).degree_v() );
        outer[4] = (*i).slice<gpucast::gl::vec3d> (gpucast::math::point3d::z, 0 );
        outer[5] = (*i).slice<gpucast::gl::vec3d> (gpucast::math::point3d::z, (*i).degree_w() );

        outer[0].tesselate(per_vertex_position, indices);
        outer[1].tesselate(per_vertex_position, indices);
        outer[2].tesselate(per_vertex_position, indices);
        outer[3].tesselate(per_vertex_position, indices);
        outer[4].tesselate(per_vertex_position, indices);
        outer[5].tesselate(per_vertex_position, indices);

        std::vector<int> junk; // already have indices from position tesselation
        for ( beziervolume::attribute_volume_map::const_iterator attrib_pair = i->data_begin(); attrib_pair != i->data_end(); ++attrib_pair)
        {
          std::array<gpucast::math::beziersurface<gpucast::gl::vec4d>, 6 > outer_attrib;

          outer_attrib[0] = attrib_pair->second.slice<gpucast::gl::vec4d> (gpucast::math::point3d::x, 0);
          outer_attrib[1] = attrib_pair->second.slice<gpucast::gl::vec4d> (gpucast::math::point3d::x, (*i).degree_u() );
          outer_attrib[2] = attrib_pair->second.slice<gpucast::gl::vec4d> (gpucast::math::point3d::y, 0 );
          outer_attrib[3] = attrib_pair->second.slice<gpucast::gl::vec4d> (gpucast::math::point3d::y, (*i).degree_v() );
          outer_attrib[4] = attrib_pair->second.slice<gpucast::gl::vec4d> (gpucast::math::point3d::z, 0 );
          outer_attrib[5] = attrib_pair->second.slice<gpucast::gl::vec4d> (gpucast::math::point3d::z, (*i).degree_w() );

          outer_attrib[0].tesselate(per_vertex_attribute_map.at(attrib_pair->first), junk);
          outer_attrib[1].tesselate(per_vertex_attribute_map.at(attrib_pair->first), junk);
          outer_attrib[2].tesselate(per_vertex_attribute_map.at(attrib_pair->first), junk);
          outer_attrib[3].tesselate(per_vertex_attribute_map.at(attrib_pair->first), junk);
          outer_attrib[4].tesselate(per_vertex_attribute_map.at(attrib_pair->first), junk);
          outer_attrib[5].tesselate(per_vertex_attribute_map.at(attrib_pair->first), junk);
        }
      }

      p.second->count = unsigned(indices.size()) - p.second->index;
    }

    // update index array buffer
    _indexarray.update ( indices.begin(), indices.end() );
    _vertices.update   ( per_vertex_position.begin(), per_vertex_position.end() );

    // copy data to attribute arraybuffer
    BOOST_FOREACH ( std::string const& attribute_name, _attributelist )
    {
      _attributes.at(attribute_name)->update ( per_vertex_attribute_map.at(attribute_name).begin(), 
                                               per_vertex_attribute_map.at(attribute_name).end());
    }
    */
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer_tesselator::_init ()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void
  volume_renderer_tesselator::_init_shader ()
  {
    volume_renderer::_init_shader();
    init_program ( _render_pass, "/volumetesselator/tesselator.vert", "/volumetesselator/tesselator.frag" );
  }


} // namespace gpucast
