/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface/fragment/fragmentlist_generator.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/fragment/fragmentlist_generator.hpp"

// header, system
#include <GL/glew.h>

#include <gpucast/gl/util/contextinfo.hpp>

//// header, project

namespace gpucast {

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  fragmentlist_generator::fragmentlist_generator( int argc, char** argv )
    : volume_renderer                 (argc, argv),
      _drawable                       (),
      _object_initialized             ( false ),
      _gl_initialized                 ( false ),
      _volume_data_texturebuffer      (),
      _volume_points_texturebuffer    (),
      _surface_data_texturebuffer     (),
      _surface_points_texturebuffer   (),
      _attribute_data_texturebuffer   (),
      _attribute_points_texturebuffer (),
      _indextexture                   (),
      _semaphoretexture               (),
      _allocation_grid                (),
      _indexlist                      (),
      _master_counter                 (),
      _discard_by_minmax              ( false ),
      _enable_sorting                 ( false ),
      _render_side_chulls             ( true ),
      _readback                       ( false ),
      _usage_indexbuffer              ( 0 ),
      _usage_fragmentbuffer           ( 0 ),
      _allocation_grid_width          ( 64 ),
      _allocation_grid_height         ( 64 ),
      _pagesize                       ( 4 ),
      _pagesize_per_fragment          ( 4 ),
      _maxsize_fragmentbuffer         ( 256000000 ),
      _maxsize_fragmentdata           ( 10000 ),
      _clean_pass                     (),
      _hull_pass                      (),
      _sort_pass                      (),
      _quad                           ()
  {
    _init();
  }


  /////////////////////////////////////////////////////////////////////////////
  fragmentlist_generator::~fragmentlist_generator()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void
  fragmentlist_generator::clear ()
  {
    volume_renderer::clear();
    _drawable.reset();
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  fragmentlist_generator::draw ()
  {
    // if there are no drawables -> return
    if ( !_object ) return;

    // initialize gl stuff 
    if ( !_gl_initialized )    
    {
      _initialize_gl_resources ();
      _gl_initialized = true;
    }

    // clear old frame
    _clear_images();

    glFinish();

    // backup state
    GLboolean cullface_enabled, depth_test_enabled;
    glGetBooleanv(GL_CULL_FACE,  &cullface_enabled);
    glGetBooleanv(GL_DEPTH_TEST, &depth_test_enabled);

    // render convexhulls to generate fragment lists
    if ( _backface_culling ) 
    {
      glEnable    ( GL_CULL_FACE );
      glFrontFace ( GL_CCW );
    } else {
      glDisable   ( GL_CULL_FACE );
    }

    glDisable ( GL_DEPTH_TEST );

    _generate_fragmentlists();

    //glFinish();


// readback fragment count / allocated memory
#if 0 
    {
      glFinish();

      std::vector<unsigned> data ( 2 * _allocation_grid_width * _allocation_grid_height );
      
      // just readback for upper left tiled index
      _allocation_grid->getbuffersubdata(0, 2 * sizeof(unsigned), &data[0]);
      glFinish();

      std::cout << "\r" << "indexlist index: " << data[0] << " fragmentdata index : " << data[0] * _pagesize_per_fragment;
      
    } 
#endif

    if ( _enable_sorting )
    {
      _sort_fragmentlists();
    }

    // restore state
    if ( !cullface_enabled ) {
      glDisable ( GL_CULL_FACE );
    } else {
      glEnable ( GL_CULL_FACE );
    }

    if ( depth_test_enabled ) {
      glEnable  ( GL_DEPTH_TEST );
    } else {
      glDisable ( GL_DEPTH_TEST );
    }

    //glFinish();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                      
  fragmentlist_generator::transform ( gpucast::gl::matrix4f const& m )
  {
    _modelmatrix = m;
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  fragmentlist_generator::compute_nearfar ()
  {
    // attributes _nearplane, _farplane
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  fragmentlist_generator::recompile ()
  {
    // clear shader
    _clean_pass.reset();
    _hull_pass.reset();
    _sort_pass.reset();

    // re-init shader
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  fragmentlist_generator::resize ( int w, int h )
  {
    volume_renderer::resize(w, h);

    _indextexture.reset         ( new gpucast::gl::texture2d );
    _semaphoretexture.reset     ( new gpucast::gl::texture2d );
    _fragmentcount.reset        ( new gpucast::gl::texture2d );

    // reallocate memory for indextexture
    _indextexture->teximage     ( 0, GL_R32UI, _width, _height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, 0 );
    _semaphoretexture->teximage ( 0, GL_R32UI, _width, _height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, 0 );
    _fragmentcount->teximage    ( 0, GL_R32UI, _width, _height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, 0 );

    // allocate memory for lists
    _indexlist->bufferdata      ( _maxsize_fragmentbuffer, 0 );
    _allocation_grid->bufferdata( _allocation_grid_height * _allocation_grid_width * sizeof(unsigned), 0 );

    // set texture format
    _indexlist->format          ( GL_RGBA32UI );
    _allocation_grid->format    ( GL_R32UI );
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::readback_information ( bool enable )
  {
    _readback = enable;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::allocation_grid_width ( unsigned width )
  {
    _allocation_grid_width = width;  

    _allocation_grid->bufferdata  ( _allocation_grid_height * _allocation_grid_width * sizeof(unsigned), 0 );
  }

  /////////////////////////////////////////////////////////////////////////////
  unsigned
  fragmentlist_generator::allocation_grid_width () const
  {
    return _allocation_grid_width;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::allocation_grid_height ( unsigned height )
  {
    _allocation_grid_height = height;  

    _allocation_grid->bufferdata  ( _allocation_grid_height * _allocation_grid_width * sizeof(unsigned), 0 );
  }

  /////////////////////////////////////////////////////////////////////////////
  unsigned
  fragmentlist_generator::allocation_grid_height () const
  {
    return _allocation_grid_height;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::pagesize ( unsigned size )
  {
    _pagesize = size;

    // clear indexlist data
    _indexlist->buffersubdata ( 0, _maxsize_fragmentbuffer, 0 );
    _indexlist->format        ( GL_RGBA32UI );
  }


  /////////////////////////////////////////////////////////////////////////////
  unsigned
  fragmentlist_generator::pagesize () const
  {
    return _pagesize;
  }

  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::readback ( )
  {
    glFinish();

    unsigned elements   = _allocation_grid_height * _allocation_grid_width;
    unsigned buffersize = elements * sizeof(unsigned);

    unsigned* data = static_cast<unsigned*>(_allocation_grid->map_range(0, buffersize, GL_MAP_READ_BIT ));

    std::vector<unsigned> page_indices ( elements );
    std::copy(data, data + elements, page_indices.begin());

    _allocation_grid->unmap();

    _usage_indexbuffer = *std::max_element(page_indices.begin(), page_indices.end());
  }


  /////////////////////////////////////////////////////////////////////////////
  unsigned                                
  fragmentlist_generator::usage_fragmentlist ( ) const
  {
    return _usage_indexbuffer;
  }


  /////////////////////////////////////////////////////////////////////////////
  unsigned                                
  fragmentlist_generator::usage_fragmentdata ( ) const
  {
    return _usage_fragmentbuffer;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  fragmentlist_generator::write ( std::ostream& os ) const
  {
    gpucast::write (os, _surface_data);
    gpucast::write (os, _surface_points);
    gpucast::write (os, _volume_data);
    gpucast::write (os, _volume_points);
    gpucast::write (os, _attribute_data);
    gpucast::write (os, _attribute_points);

    gpucast::write (os, _vertices);
    gpucast::write (os, _vertexparameter);

    _renderinfo.write(os);
  }


  /////////////////////////////////////////////////////////////////////////////
  void                          
  fragmentlist_generator::read ( std::istream& is )
  {
    gpucast::read (is, _surface_data);
    gpucast::read (is, _surface_points);
    gpucast::read (is, _volume_data);
    gpucast::read (is, _volume_points);
    gpucast::read (is, _attribute_data);
    gpucast::read (is, _attribute_points);

    gpucast::read (is, _vertices);
    gpucast::read (is, _vertexparameter);

    _renderinfo.read(is);

    _object_initialized = true;
    _gl_initialized     = false;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  fragmentlist_generator::_initialize_gl_resources ()
  {
    if ( !_drawable ) {
      _drawable.reset ( new drawable_ressource_impl );
    }

    _upload_vertexarrays();

    _upload_indexarrays();

    _initialize_texbuffers();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::_upload_indexarrays ()
  {
    _drawable->indexarray.update   ( _renderinfo.begin(), _renderinfo.end() );
    _drawable->size = unsigned     ( _renderinfo.size() );

    std::cout << "Proxy geometry: " << (_renderinfo.size() / 3) << " triangles" << std::endl;
    std::cout << "Allocating " << (sizeof(int) * _renderinfo.size() / 1024) << " kB for indexarray" <<std::endl;
  }
  

  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::_upload_vertexarrays ()
  {
    if ( !_vertices.empty() )
    {
      std::cout << "Allocating " << sizeof(gpucast::gl::vec3f) * std::distance(_vertices.begin(),        _vertices.end()) / 1024 << " kB for attribarray0" << std::endl;
      std::cout << "Allocating " << sizeof(gpucast::gl::vec4f) * std::distance(_vertexparameter.begin(), _vertexparameter.end()) / 1024 << " kB for attribarray1" <<std::endl;

      // copy vertex and attribute data to gpu
      _drawable->attribarray0.update ( _vertices.begin(),         _vertices.end() );
      _drawable->attribarray1.update ( _vertexparameter.begin(),  _vertexparameter.end() );

      // initialize vao's
      _initialize_vao();
    } 
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::_initialize_vao ( )
  {
    // bind vertex array object and reset vertexarrayoffsets
    _drawable->vao.bind(); 

    _drawable->vao.attrib_array ( _drawable->attribarray0, 0, 3, GL_FLOAT, false, 0, 0 );
    _drawable->vao.enable_attrib(0);

    _drawable->vao.attrib_array ( _drawable->attribarray1, 1, 4, GL_FLOAT, false, 0, 0 );
    _drawable->vao.enable_attrib(1);

    // finally unbind vertex array object
    _drawable->vao.unbind();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                    
  fragmentlist_generator::_initialize_texbuffers ()
  {
    // update texture buffer
    _surface_data_texturebuffer.update     ( _surface_data.begin(),     _surface_data.end()  );
    _surface_points_texturebuffer.update   ( _surface_points.begin(),   _surface_points.end()  );
    _volume_data_texturebuffer.update      ( _volume_data.begin(),      _volume_data.end()  );
    _volume_points_texturebuffer.update    ( _volume_points.begin(),    _volume_points.end()  );
    _attribute_data_texturebuffer.update   ( _attribute_data.begin(),   _attribute_data.end()  );
    _attribute_points_texturebuffer.update ( _attribute_points.begin(), _attribute_points.end()  );

    _surface_data_texturebuffer.format     ( GL_RGBA32UI );
    _surface_points_texturebuffer.format   ( GL_RGBA32F );
    _volume_data_texturebuffer.format      ( GL_RGBA32F );
    _volume_points_texturebuffer.format    ( GL_RGBA32F );
    _attribute_data_texturebuffer.format   ( GL_RGBA32F );
    _attribute_points_texturebuffer.format ( GL_RG32F );

    std::size_t srfdata    = std::distance ( _surface_data.begin(), _surface_data.end());
    std::size_t srfpoints  = std::distance ( _surface_points.begin(), _surface_points.end());
    std::size_t voldata    = std::distance ( _volume_data.begin(),  _volume_data.end());
    std::size_t volpoints  = std::distance ( _volume_points.begin(),  _volume_points.end());
    std::size_t attrdata   = std::distance ( _attribute_data.begin(), _attribute_data.end());
    std::size_t attrpoints = std::distance ( _attribute_points.begin(),  _attribute_points.end());

    std::cout << "Allocating : " << srfdata    * sizeof(gpucast::gl::vec4u) / 1024 << " kB for surface data" << std::endl;
    std::cout << "Allocating : " << srfpoints  * sizeof(gpucast::gl::vec4f) / 1024 << " kB for surface points" << std::endl;
    std::cout << "Allocating : " << voldata    * sizeof(gpucast::gl::vec4f) / 1024 << " kB for volume data" << std::endl;
    std::cout << "Allocating : " << volpoints  * sizeof(gpucast::gl::vec4f) / 1024 << " kB for volume points" << std::endl;
    std::cout << "Allocating : " << attrdata   * sizeof(gpucast::gl::vec4u) / 1024 << " kB for attribute data" << std::endl;
    std::cout << "Allocating : " << attrpoints * sizeof(gpucast::gl::vec2f) / 1024 << " kB for attribute points" << std::endl;

    gpucast::gl::print_memory_usage ( std::cout );
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  fragmentlist_generator::_init ()
  {
    // init resources
    _indextexture.reset     ( new gpucast::gl::texture2d );
    _semaphoretexture.reset ( new gpucast::gl::texture2d );
    _fragmentcount.reset    ( new gpucast::gl::texture2d );

    _allocation_grid.reset  ( new gpucast::gl::texturebuffer( _allocation_grid_width * _allocation_grid_height * sizeof(unsigned), GL_DYNAMIC_COPY, GL_R32UI ) );
    _indexlist.reset        ( new gpucast::gl::texturebuffer( _maxsize_fragmentbuffer, GL_DYNAMIC_COPY, GL_RGBA32UI ) );
    _master_counter.reset   ( new gpucast::gl::texturebuffer( 2 * sizeof(unsigned), GL_DYNAMIC_COPY, GL_R32UI ) );

    _quad.reset             ( new gpucast::gl::plane(0, -1, 1) );

    // init shaders
    _init_shader();
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  fragmentlist_generator::_init_shader ( )
  {
    volume_renderer::_init_shader();

    if ( !_clean_pass) init_program ( _clean_pass,   "/volumefragmentgenerator/clean_pass.vert", "/volumefragmentgenerator/clean_pass.frag" );
    if ( !_hull_pass)  init_program ( _hull_pass,    "/volumefragmentgenerator/hull_pass.vert",  "/volumefragmentgenerator/hull_pass.frag", "/volumefragmentgenerator/hull_pass.geom" );
    if ( !_sort_pass)  init_program ( _sort_pass,    "/volumefragmentgenerator/sort_pass.vert",  "/volumefragmentgenerator/sort_pass.frag" );
  } 


  /////////////////////////////////////////////////////////////////////////////
  void                                
  fragmentlist_generator::_clear_images ()
  {
    // clear index buffer
    _clean_pass->begin();
    {
      _clean_pass->set_uniform1i        ("pagesize",              _pagesize );
      _clean_pass->set_uniform1i        ("pagesize_per_fragment", _pagesize_per_fragment );
      _clean_pass->set_uniform1i        ("width",                 _width );
      _clean_pass->set_uniform1i        ("height",                _height );
      _clean_pass->set_uniform2i        ("tile_size",             _allocation_grid_width, _allocation_grid_height );

      _indextexture->bind_as_image      ( 0, 0, false, 0, GL_WRITE_ONLY, GL_R32UI );
      _clean_pass->set_uniform1i        ("index_image", 0);

      _allocation_grid->bind_as_image   ( 1, 0, false, 0, GL_WRITE_ONLY, GL_R32UI );
      _clean_pass->set_uniform1i        ("globalindices", 1);

      _semaphoretexture->bind_as_image  ( 2, 0, false, 0, GL_WRITE_ONLY, GL_R32UI );
      _clean_pass->set_uniform1i        ("semaphore_image", 2);

      _indexlist->bind_as_image         ( 3, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32UI );
      _clean_pass->set_uniform1i        ("indexlist", 3);

      _fragmentcount->bind_as_image     ( 4, 0, false, 0, GL_WRITE_ONLY, GL_R32UI );
      _clean_pass->set_uniform1i        ("fragmentcount", 4);

      _master_counter->bind_as_image    ( 5, 0, false, 0, GL_WRITE_ONLY, GL_R32UI );
      _clean_pass->set_uniform1i        ("master_counter", 5);

      _quad->draw();
    }
    _clean_pass->end();
 }


  /////////////////////////////////////////////////////////////////////////////
  void                                
  fragmentlist_generator::_generate_fragmentlists ()
  {
    _hull_pass->begin();
    {
      _hull_pass->set_uniform1f         ("nearplane",                            _nearplane );
      _hull_pass->set_uniform1f         ("farplane",                             _farplane );
      _hull_pass->set_uniform1i         ("width",                                _width );
      _hull_pass->set_uniform1i         ("height",                               _height );
      _hull_pass->set_uniform1i         ("pagesize",                             _pagesize );
      _hull_pass->set_uniform1i         ("pagesize_per_fragment",                _pagesize_per_fragment );
      _hull_pass->set_uniform2i         ("tile_size",                            _allocation_grid_width, _allocation_grid_height );
      _hull_pass->set_uniform1i         ("discard_by_minmax",                    _discard_by_minmax );

      float attribute_minimum     = _global_attribute_bounds.minimum();
      float attribute_maximum     = _global_attribute_bounds.maximum();
      float attribute_range       = attribute_maximum - attribute_minimum;
      float normalized_threshold  = attribute_minimum + attribute_range * _relative_isovalue;

      _hull_pass->set_uniform1f        ("threshold",                normalized_threshold);

      gpucast::gl::matrix4f modelview        = _modelviewmatrix * _modelmatrix;
      gpucast::gl::matrix4f modelviewinverse = gpucast::gl::inverse(modelview);
      gpucast::gl::matrix4f normalmatrix     = modelview.normalmatrix();
      gpucast::gl::matrix4f mvp              = _projectionmatrix * modelview;
      gpucast::gl::matrix4f mvp_inverse      = gpucast::gl::inverse(mvp);

      _hull_pass->set_uniform_matrix4fv ("normalmatrix",              1, false, &normalmatrix[0] );
      _hull_pass->set_uniform_matrix4fv ("modelviewmatrix",           1, false, &modelview[0] );
      _hull_pass->set_uniform_matrix4fv ("modelviewmatrixinverse",    1, false, &modelviewinverse[0] );
      _hull_pass->set_uniform_matrix4fv ("modelviewprojectionmatrix", 1, false, &mvp[0] );

      int texunit = 0;
      _hull_pass->set_texturebuffer("volumedatabuffer",      _volume_data_texturebuffer,       texunit++ );
      _hull_pass->set_texturebuffer("surfacedatabuffer",     _surface_data_texturebuffer,      texunit++ );
      _hull_pass->set_texturebuffer("attributedatabuffer",   _attribute_data_texturebuffer,    texunit++ );

      _indextexture->bind_as_image( texunit, 0, false, 0, GL_READ_WRITE, GL_R32UI );
      _hull_pass->set_uniform1i("index_image", texunit++);

      _semaphoretexture->bind_as_image( texunit, 0, false, 0, GL_READ_WRITE, GL_R32UI );
      _hull_pass->set_uniform1i("semaphore_image", texunit++);

      _indexlist->bind_as_image( texunit, 0, false, 0, GL_READ_WRITE, GL_RGBA32UI );
      _hull_pass->set_uniform1i("indexlist", texunit++);

      _allocation_grid->bind_as_image( texunit, 0, false, 0, GL_READ_WRITE, GL_R32UI );
      _hull_pass->set_uniform1i("globalindices", texunit++);

      _fragmentcount->bind_as_image( texunit, 0, false, 0, GL_READ_WRITE, GL_R32UI );
      _hull_pass->set_uniform1i("fragmentcount", texunit++);

      _master_counter->bind_as_image( texunit, 0, false, 0, GL_READ_WRITE, GL_R32UI );
      _hull_pass->set_uniform1i("master_counter", texunit++);

      _drawable->vao.bind();
      {
        // bind index array and set draw pointer
        _drawable->indexarray.bind();
        {
          int outer_base  = 0;
          int outer_count = 0;
          int inner_base  = 0;
          int inner_count = 0;

          // draw outer surface bounding geometry
          _renderinfo.get_outerbin ( outer_base, outer_count );

          //glDrawElements ( GL_TRIANGLES, count, GL_UNSIGNED_INT, (GLvoid*)(base*sizeof(unsigned)));
          glDrawRangeElements ( GL_TRIANGLES, outer_base, outer_base+outer_count, outer_count, GL_UNSIGNED_INT, (GLvoid*)(outer_base*sizeof(unsigned)) );

          // draw inner surface elements
          if ( _renderinfo.get_renderbin ( normalized_threshold, inner_base, inner_count ) ) 
          {
            //glDrawElements            ( GL_TRIANGLES, count, GL_UNSIGNED_INT, (GLvoid*)(base*sizeof(unsigned)) );
            glDrawRangeElements       ( GL_TRIANGLES, inner_base, inner_base+inner_count, inner_count, GL_UNSIGNED_INT, (GLvoid*)(inner_base*sizeof(unsigned)) );
          }

          std::cout << "tris drawn : " << (inner_count + outer_count)/3 << " ";
        }
        _drawable->indexarray.unbind();
      }
      _drawable->vao.unbind();
    }
    _hull_pass->end();
  }


  /////////////////////////////////////////////////////////////////////////////
  void                                
  fragmentlist_generator::_sort_fragmentlists ()
  {
    // clear index buffer
    _sort_pass->begin();
    {
      _sort_pass->set_uniform1i        ("width",                 _width );
      _sort_pass->set_uniform1i        ("height",                _height );
      _sort_pass->set_uniform1i        ("pagesize",              _pagesize );
      _sort_pass->set_uniform2i        ("tile_size",             _allocation_grid_width, _allocation_grid_height );

      _indexlist->bind_as_image        ( 0, 0, false, 0, GL_READ_WRITE, GL_RGBA32UI );
      _sort_pass->set_uniform1i        ( "indexlist", 0 );

      _fragmentcount->bind_as_image    ( 1, 0, false, 0, GL_READ_ONLY, GL_R32UI );
      _sort_pass->set_uniform1i        ( "fragmentcount", 1 );

      _quad->draw();
    }
    _sort_pass->end();
  }

} // namespace gpucast
