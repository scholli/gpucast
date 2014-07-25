/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : surface_renderer_gl.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/surface_renderer_gl.hpp"

// header, system
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/get_nearfar.hpp>

// header, project
#include <gpucast/core/beziersurfaceobject.hpp>

namespace gpucast {

/////////////////////////////////////////////////////////////////////////////
// per drawble ressource
/////////////////////////////////////////////////////////////////////////////
struct surface_renderer_gl::drawable_ressource_impl
{
  drawable_ressource_impl()
    : size (0)
  {}

  std::size_t                        size;

  gpucast::gl::arraybuffer           attribarray0;
  gpucast::gl::arraybuffer           attribarray1;
  gpucast::gl::arraybuffer           attribarray2;
  gpucast::gl::arraybuffer           attribarray3;

  gpucast::gl::elementarraybuffer    indexarray;

  gpucast::gl::texturebuffer         controlpoints;

  // contour map
  gpucast::gl::texturebuffer         cmb_partition;
  gpucast::gl::texturebuffer         cmb_contourlist;
  gpucast::gl::texturebuffer         cmb_curvelist;
  gpucast::gl::texturebuffer         cmb_curvedata;
  gpucast::gl::texturebuffer         cmb_pointdata;

  // double binary
  gpucast::gl::texturebuffer         db_partition;
  gpucast::gl::texturebuffer         db_celldata;
  gpucast::gl::texturebuffer         db_curvelist;
  gpucast::gl::texturebuffer         db_curvedata;

  gpucast::gl::vertexarrayobject     vao;
};
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  surface_renderer_gl::surface_renderer_gl( int argc, char** argv )
    : surface_renderer( argc, argv ),
      _drawables      (),
      _program        (),
      _cubemap        (),
      _spheremap      (),
      _diffusemap     ()
  {
    _cubemap.reset      ( new gpucast::gl::cubemap );
    _spheremap.reset    ( new gpucast::gl::texture2d );
    _diffusemap.reset   ( new gpucast::gl::texture2d );
    _linear_interp.reset( new gpucast::gl::sampler );
  }


  /////////////////////////////////////////////////////////////////////////////
  surface_renderer_gl::~surface_renderer_gl()
  {}


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ surface_renderer::drawable_ptr
  surface_renderer_gl::create ()
  {
    drawable_ptr bezierdrawable ( new drawable_type );
    _drawables.insert(std::make_pair(bezierdrawable, drawable_ressource_ptr(new drawable_ressource_impl) ));
    return bezierdrawable;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                          
  surface_renderer_gl::clear ()
  {
    _drawables.clear();
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void 
  surface_renderer_gl::draw ()
  {
    if (!_program) {
      _init_shader();
    }
    _program->begin();

    // set global program state
    _program->set_uniform1i         ("iterations", GLint   ( _newton_iterations ));
    _program->set_uniform1f         ("nearplane",  GLfloat ( _nearplane         ));
    _program->set_uniform1f         ("farplane",   GLfloat ( _farplane          ));
    _program->set_uniform1f         ("epsilon_object_space", GLfloat(0.001f));

    _program->set_cubemap           ("cubemap",     *_cubemap,    3);
    _program->set_texture2d         ("spheremap",   *_spheremap,  4);
    _program->set_texture2d         ("diffusemap",  *_diffusemap, 5);

    _linear_interp->bind(3);
    _linear_interp->bind(4);
    _linear_interp->bind(5);

    _program->set_uniform1i         ("cubemapping",         int(_cube_mapping) );
    _program->set_uniform1i         ("spheremapping",       int(_sphere_mapping) );
    _program->set_uniform1i         ("diffusemapping",      int(_diffuse_mapping) );
    _program->set_uniform1i         ("trimapproach",        int(_trimapproach) );
    _program->set_uniform1i         ("trimming_enabled",    _trimming_enabled);
    _program->set_uniform1i         ("raycasting_enabled",  _raycasting_enabled);
    _program->set_uniform4f         ("lightpos0",           0.0f, 100.0f, -100.0f, 1.0f);

    _program->set_uniform_matrix4fv ("normalmatrix",              1, false, &_normalmatrix[0]);
    _program->set_uniform_matrix4fv ("modelviewmatrix",           1, false, &_modelviewmatrix[0]);
    _program->set_uniform_matrix4fv ("modelviewmatrixinverse",    1, false, &_modelviewmatrixinverse[0]);
    _program->set_uniform_matrix4fv ("modelviewprojectionmatrix", 1, false, &_modelviewprojectionmatrix[0]);

    // first of all update all objects
    for (drawable_ressource_pair const& p : _drawables) 
    {
      if (!p.first->initialized()) 
      {
        p.first->init();    // initialize client side data
        _sync(p);           // synchronize with GPU memory
      }
    }

    // actually draw stuff
    for (drawable_ressource_pair const& p : _drawables) 
    {
      // set per-object state
      _program->set_uniform3f    ("mat_ambient" , 0.1f, 0.1f, 0.1f);
      _program->set_uniform3f    ("mat_diffuse",  0.7f, 0.5f, 0.1f);
      _program->set_uniform3f    ("mat_specular", 0.5f, 0.5f, 0.5f);
      _program->set_uniform1f    ("shininess",    0.3f);
      _program->set_uniform1f    ("opacity",      1.0f);

      unsigned texunit = 6;
      _program->set_texturebuffer("vertexdata",      p.second->controlpoints, texunit++);

      _program->set_texturebuffer("cmb_partition",   p.second->cmb_partition,    texunit++);
      _program->set_texturebuffer("cmb_contourlist", p.second->cmb_contourlist,   texunit++);
      _program->set_texturebuffer("cmb_curvelist",   p.second->cmb_contourlist,   texunit++);
      _program->set_texturebuffer("cmb_curvedata",   p.second->cmb_curvelist,     texunit++);
      _program->set_texturebuffer("cmb_pointdata",   p.second->cmb_pointdata,     texunit++);
                                                                              
      _program->set_texturebuffer("bp_trimdata",     p.second->db_partition,     texunit++);
      _program->set_texturebuffer("bp_celldata",     p.second->db_celldata,      texunit++);
      _program->set_texturebuffer("bp_curvelist",    p.second->db_curvelist,     texunit++);
      _program->set_texturebuffer("bp_curvedata",    p.second->db_curvedata,     texunit++);

      p.second->vao.bind();
      {
        // bind index array and set draw pointer
        p.second->indexarray.bind();
        {
          glDrawElements(GL_TRIANGLES, GLsizei(p.second->size), GL_UNSIGNED_INT, 0);
        }
        p.second->indexarray.unbind();
      }
      p.second->vao.unbind();
    }
    _program->end();
  }

#if 1
  /////////////////////////////////////////////////////////////////////////////
  void 
  surface_renderer_gl::draw ( drawable_ptr const& drawable )
  {
    if (!_program) {
      _init_shader();
    }

    drawable_map::iterator object = _drawables.find(drawable);
    if ( object == _drawables.end()) {
      return;
    } else 
    {

      _program->begin();

      unsigned texunit = 0;

      // set global program state
      _program->set_uniform1i         ("iterations", GLint   ( _newton_iterations ));
      _program->set_uniform1f         ("nearplane",  GLfloat ( _nearplane         ));
      _program->set_uniform1f         ("farplane",   GLfloat ( _farplane          ));
      _program->set_uniform1f         ("epsilon_object_space", GLfloat(0.001f));

      _program->set_cubemap           ("cubemap",     *_cubemap,    texunit);
      _linear_interp->bind(texunit++);
      _program->set_texture2d         ("spheremap",   *_spheremap,  texunit);
      _linear_interp->bind(texunit++);
      _program->set_texture2d         ("diffusemap",  *_diffusemap, texunit);
      _linear_interp->bind(texunit++);

      _program->set_uniform1i         ("cubemapping",         _cube_mapping );
      _program->set_uniform1i         ("spheremapping",       _sphere_mapping );
      _program->set_uniform1i         ("diffusemapping",      _diffuse_mapping );
      _program->set_uniform1i         ("trimapproach",        int(_trimapproach) );

      _program->set_uniform1i         ("trimming_enabled",    _trimming_enabled);
      _program->set_uniform1i         ("raycasting_enabled",  _raycasting_enabled);

      _program->set_uniform_matrix4fv ("normalmatrix",              1, false, &_normalmatrix[0]);
      _program->set_uniform_matrix4fv ("modelviewmatrix",           1, false, &_modelviewmatrix[0]);
      _program->set_uniform_matrix4fv ("modelviewmatrixinverse",    1, false, &_modelviewmatrixinverse[0]);
      _program->set_uniform_matrix4fv ("modelviewprojectionmatrix", 1, false, &_modelviewprojectionmatrix[0]);
      // first of all update all objects

      if (!object->first->initialized()) 
      {
        object->first->init();
      }

      if (object->second->size == 0)
      {
        _sync(*object);           // synchronize with GPU memory
      }

      // set per-object state
      _program->set_uniform3f         ("mat_ambient", 0.1f, 0.1f, 0.1f);
      _program->set_uniform3f         ("mat_diffuse", 0.7f, 0.5f, 0.1f);
      _program->set_uniform3f         ("mat_specular", 0.5f, 0.5f, 0.5f);
      _program->set_uniform1f         ("shininess",    0.5f);
      _program->set_uniform1f         ("opacity",      1.0f);

      _program->set_texturebuffer     ("vertexdata",   object->second->controlpoints,  texunit++);

      _program->set_texturebuffer     ("cmb_partition",  object->second->cmb_partition,  texunit++);
      _program->set_texturebuffer     ("cmb_contourlist", object->second->cmb_contourlist, texunit++);
      _program->set_texturebuffer     ("cmb_curvelist",   object->second->cmb_curvelist,   texunit++);
      _program->set_texturebuffer     ("cmb_curvedata",   object->second->cmb_curvedata,   texunit++);
      _program->set_texturebuffer     ("cmb_pointdata",   object->second->cmb_pointdata,   texunit++);

      _program->set_texturebuffer     ("bp_trimdata",     object->second->db_partition,   texunit++);
      _program->set_texturebuffer     ("bp_celldata",     object->second->db_celldata,    texunit++);
      _program->set_texturebuffer     ("bp_curvelist",    object->second->db_curvelist,   texunit++);
      _program->set_texturebuffer     ("bp_curvedata",    object->second->db_curvedata,   texunit++);

      object->second->vao.bind();
      {
        // bind index array and set draw pointer
        object->second->indexarray.bind();
        {
          glDrawElements(GL_TRIANGLES, GLsizei(object->second->size), GL_UNSIGNED_INT, 0);
        }
        object->second->indexarray.unbind();
      }
      object->second->vao.unbind();

      _program->end();
    }
  }
#endif

  /////////////////////////////////////////////////////////////////////////////
  void                            
  surface_renderer_gl::memory_usage( std::size_t& trim_data_binarypartition_bytes,
                                     std::size_t& trim_data_contourmap_bytes,
                                     std::size_t& surface_data_bytes ) const
  {
    for (drawable_ressource_pair const& p : _drawables) 
    {
      surface_data_bytes += p.first->_attrib0.size() * sizeof ( gpucast::gl::vec4f ) + 
                            p.first->_attrib1.size() * sizeof ( gpucast::gl::vec4f ) + 
                            p.first->_attrib2.size() * sizeof ( gpucast::gl::vec4f ) + 
                            p.first->_attrib3.size() * sizeof ( gpucast::gl::vec4f ) + 
                            p.first->_indices.size() * sizeof ( int );// + 
                            //p.first->_controlpoints.size() * sizeof ( gpucast::gl::vec4f );

      trim_data_contourmap_bytes += p.first->_cmb_partition.size()  * sizeof ( gpucast::gl::vec4f ) +
                                    p.first->_cmb_contourlist.size() * sizeof ( gpucast::gl::vec2f ) +
                                    p.first->_cmb_curvelist.size()   * sizeof ( gpucast::gl::vec4f ) +
                                    p.first->_cmb_curvedata.size()   * sizeof ( float );// +
                                    //p.first->_cmb_pointdata.size()   * sizeof ( gpucast::gl::vec3f );

      trim_data_binarypartition_bytes += p.first->_db_partition.size()  * sizeof ( gpucast::gl::vec4f ) +
                                         p.first->_db_celldata.size()   * sizeof ( gpucast::gl::vec4f ) +
                                         p.first->_db_curvelist.size()  * sizeof ( gpucast::gl::vec4f ) +
                                         p.first->_db_curvedata.size()  * sizeof ( gpucast::gl::vec3f );
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void            
  surface_renderer_gl::spheremap  ( std::string const& filepath )
  {
    _spheremap->load(filepath);
  }

  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void            
  surface_renderer_gl::diffusemap  ( std::string const& filepath )
  {
    _diffusemap->load(filepath);
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void            
  surface_renderer_gl::cubemap ( std::string const& positive_x,
                                std::string const& negative_x,
                                std::string const& positive_y,
                                std::string const& negative_y,
                                std::string const& positive_z,
                                std::string const& negative_z )
  {
    _cubemap->load(positive_x, negative_x, positive_y, negative_y, positive_z, negative_z);
  }


  /////////////////////////////////////////////////////////////////////////////
  void                          
  surface_renderer_gl::recompile ()
  {
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                          
  surface_renderer_gl::_sync ( drawable_ressource_pair const& p)
  {
    // update attribute buffers
    p.second->attribarray0.update(p.first->_attrib0.begin(), p.first->_attrib0.end()); 
    p.second->attribarray1.update(p.first->_attrib1.begin(), p.first->_attrib1.end());
    p.second->attribarray2.update(p.first->_attrib2.begin(), p.first->_attrib2.end());
    p.second->attribarray3.update(p.first->_attrib3.begin(), p.first->_attrib3.end());

    // update index array buffer
    p.second->indexarray.update(p.first->_indices.begin(), p.first->_indices.end());
    p.second->size = p.first->_indices.size();

    // update texturebuffers
    p.second->controlpoints.update(p.first->_controlpoints.begin(), p.first->_controlpoints.end());
    p.second->controlpoints.format(GL_RGBA32F_ARB);

    p.second->cmb_partition.update   (p.first->_cmb_partition.begin(),   p.first->_cmb_partition.end());
    p.second->cmb_contourlist.update (p.first->_cmb_contourlist.begin(), p.first->_cmb_contourlist.end());
    p.second->cmb_curvelist.update   (p.first->_cmb_curvelist.begin(),   p.first->_cmb_curvelist.end());
    p.second->cmb_curvedata.update   (p.first->_cmb_curvedata.begin(),   p.first->_cmb_curvedata.end());
    p.second->cmb_pointdata.update   (p.first->_cmb_pointdata.begin(),   p.first->_cmb_pointdata.end());


    p.second->cmb_partition.format  (GL_RGBA32F);
    p.second->cmb_contourlist.format (GL_RG32F);
    p.second->cmb_curvelist.format   (GL_RGBA32F);
    p.second->cmb_curvedata.format   (GL_R32F);
    p.second->cmb_pointdata.format   (GL_RGB32F);

    p.second->db_partition.update (p.first->_db_partition.begin(), p.first->_db_partition.end());
    p.second->db_celldata.update  (p.first->_db_celldata.begin(),  p.first->_db_celldata.end());
    p.second->db_curvelist.update (p.first->_db_curvelist.begin(), p.first->_db_curvelist.end());
    p.second->db_curvedata.update (p.first->_db_curvedata.begin(), p.first->_db_curvedata.end());
                                            
    p.second->db_partition.format (GL_RGBA32F);
    p.second->db_celldata.format  (GL_RGBA32F);
    p.second->db_curvelist.format (GL_RGBA32F);
    p.second->db_curvedata.format (GL_RGB32F);

    // bind vertex array object and reset vertexarrayoffsets
    p.second->vao.bind();

    p.second->vao.attrib_array(p.second->attribarray0, 0, 3, GL_FLOAT, false, 0, 0);
    p.second->vao.enable_attrib(0);

    p.second->vao.attrib_array(p.second->attribarray1, 1, 4, GL_FLOAT, false, 0, 0);
    p.second->vao.enable_attrib(1);

    p.second->vao.attrib_array(p.second->attribarray2, 2, 4, GL_FLOAT, false, 0, 0);
    p.second->vao.enable_attrib(2);

    p.second->vao.attrib_array(p.second->attribarray3, 3, 4, GL_FLOAT, false, 0, 0);
    p.second->vao.enable_attrib(3);

    // finally unbind vertex array object 
    p.second->vao.unbind();
  }


  /////////////////////////////////////////////////////////////////////////////
  void 
  surface_renderer_gl::_init_shader ()
  {
    init_program(_program, "./gpucast_core/glsl/trimmed_surface/raycast_surface.glsl.vert", "./gpucast_core/glsl/trimmed_surface/raycast_surface.glsl.frag");
  }

} // namespace gpucast
