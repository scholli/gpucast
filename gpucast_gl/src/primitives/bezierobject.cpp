/********************************************************************************
*
* Copyright (C) 2014 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bezierobject.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/primitives/bezierobject.hpp"

#include <fstream>
#include <regex>

#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/util/resource_factory.hpp>

#include <gpucast/math/util/prefilter2d.hpp>

#include <gpucast/core/config.hpp>

#include <boost/filesystem.hpp>

namespace gpucast {
  namespace gl {

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::bezierobject(gpucast::beziersurfaceobject const& b)
    {
      _material.randomize();
      _upload(b);
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::~bezierobject()
    {}

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::draw()
    {
      auto& renderer = bezierobject_renderer::instance();

      renderer.bind();

      renderer.apply_uniforms();
      _apply_uniforms(renderer.get_program());

      // draw proxy geometry
      _vao.bind();
      _indexarray.bind();
      glDrawElements(GL_TRIANGLES, GLsizei(_size), GL_UNSIGNED_INT, 0);
      _indexarray.unbind();
      _vao.unbind();

      renderer.unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::max_newton_iterations(unsigned n)
    {
      _iterations = n;
    }

    /////////////////////////////////////////////////////////////////////////////
    unsigned bezierobject::max_newton_iterations() const
    {
      return _iterations;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::newton_epsilon(float epsilon)
    {
      _epsilon = epsilon;
    }

    /////////////////////////////////////////////////////////////////////////////
    float bezierobject::newton_epsilon() const
    {
      return _epsilon;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::culling(bool enable)
    {
      _culling = enable;
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject::culling() const
    {
      return _culling;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::raycasting(bool enable)
    {
      _raycasting = enable;
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject::raycasting() const
    {
      return _raycasting;
    }

    /////////////////////////////////////////////////////////////////////////////
    beziersurfaceobject::trim_approach_t bezierobject::trimming() const
    {
      return _trimming;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::trimming(beziersurfaceobject::trim_approach_t type)
    {
      _trimming = type;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::antialiasing(bezierobject::anti_aliasing_mode mode)
    {
      _antialiasing = mode;
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::anti_aliasing_mode bezierobject::antialiasing() const
    {
      return _antialiasing;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::set_material(material const& m)
    {
      _material = m;
    }

    /////////////////////////////////////////////////////////////////////////////
    material const& bezierobject::get_material() const
    {
      return _material;
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_apply_uniforms(program const& p)
    {
      // render parameters
      p.set_uniform1i("iterations", _iterations);
      p.set_uniform1i("trimming", int(_trimming) );
      p.set_uniform1i("raycasting_enabled", _raycasting);
      p.set_uniform1f("epsilon_object_space", _epsilon);

      // material properties
      p.set_uniform3f("mat_ambient", _material.ambient[0], _material.ambient[1], _material.ambient[2]);
      p.set_uniform3f("mat_diffuse", _material.diffuse[0], _material.diffuse[1], _material.diffuse[2]);
      p.set_uniform3f("mat_specular", _material.specular[0], _material.specular[1], _material.specular[2]);
      p.set_uniform1f("shininess", _material.shininess);
      p.set_uniform1f("opacity", _material.opacity);

      // data uniforms
      auto& renderer = bezierobject_renderer::instance();
      p.set_texturebuffer("vertexdata", _controlpoints, renderer.next_texunit());

      p.set_texturebuffer("bp_trimdata", _db_partition, renderer.next_texunit());
      p.set_texturebuffer("bp_celldata", _db_celldata, renderer.next_texunit());
      p.set_texturebuffer("bp_curvelist", _db_curvelist, renderer.next_texunit());
      p.set_texturebuffer("bp_curvedata", _db_curvedata, renderer.next_texunit());

      p.set_texturebuffer("cmb_partition", _cmb_partition, renderer.next_texunit());
      p.set_texturebuffer("cmb_contourlist", _cmb_contourlist, renderer.next_texunit());
      p.set_texturebuffer("cmb_curvelist", _cmb_curvelist, renderer.next_texunit());
      p.set_texturebuffer("cmb_curvedata", _cmb_curvedata, renderer.next_texunit());
      p.set_texturebuffer("cmb_pointdata", _cmb_pointdata, renderer.next_texunit());
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload(gpucast::beziersurfaceobject const& b)
    {
      // update attribute buffers
      _attribarray0.update(b._attrib0.begin(), b._attrib0.end());
      _attribarray1.update(b._attrib1.begin(), b._attrib1.end());
      _attribarray2.update(b._attrib2.begin(), b._attrib2.end());
      _attribarray3.update(b._attrib3.begin(), b._attrib3.end());

      // update index array buffer
      _indexarray.update(b._indices.begin(), b._indices.end());
      _size = b._indices.size();

      // update texturebuffers
      _controlpoints.update(b._controlpoints.begin(), b._controlpoints.end());
      _controlpoints.format(GL_RGBA32F);

      _cmb_partition.update(b._cmb_partition.begin(), b._cmb_partition.end());
      _cmb_contourlist.update(b._cmb_contourlist.begin(), b._cmb_contourlist.end());
      _cmb_curvelist.update(b._cmb_curvelist.begin(), b._cmb_curvelist.end());
      _cmb_curvedata.update(b._cmb_curvedata.begin(), b._cmb_curvedata.end());
      _cmb_pointdata.update(b._cmb_pointdata.begin(), b._cmb_pointdata.end());

      _cmb_partition.format(GL_RGBA32F);
      _cmb_contourlist.format(GL_RGBA32F);
      _cmb_curvelist.format(GL_RGBA32F);
      _cmb_curvedata.format(GL_R32F);
      _cmb_pointdata.format(GL_RGB32F);

      _db_partition.update(b._db_partition.begin(), b._db_partition.end());
      _db_celldata.update(b._db_celldata.begin(), b._db_celldata.end());
      _db_curvelist.update(b._db_curvelist.begin(), b._db_curvelist.end());
      _db_curvedata.update(b._db_curvedata.begin(), b._db_curvedata.end());

      _db_partition.format(GL_RGBA32F);
      _db_celldata.format(GL_RGBA32F);
      _db_curvelist.format(GL_RGBA32F);
      _db_curvedata.format(GL_RGB32F);

      // bind vertex array object and reset vertexarrayoffsets
      _vao.bind();
      {
        _vao.attrib_array(_attribarray0, 0, 3, GL_FLOAT, false, 0, 0);
        _vao.enable_attrib(0);

        _vao.attrib_array(_attribarray1, 1, 4, GL_FLOAT, false, 0, 0);
        _vao.enable_attrib(1);

        _vao.attrib_array(_attribarray2, 2, 4, GL_FLOAT, false, 0, 0);
        _vao.enable_attrib(2);

        _vao.attrib_array(_attribarray3, 3, 4, GL_FLOAT, false, 0, 0);
        _vao.enable_attrib(3);
      }
      // finally unbind vertex array object 
      _vao.unbind();
    }
  


    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////
    bezierobject_renderer::bezierobject_renderer() 
    {
      _pathlist.insert("");

      _init_program();
      _init_prefilter();
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject_renderer::~bezierobject_renderer()
    {}

    /////////////////////////////////////////////////////////////////////////////
    bezierobject_renderer& bezierobject_renderer::instance()
    {
      static bezierobject_renderer instance;
      return instance;
    }

    /////////////////////////////////////////////////////////////////////////////
    int bezierobject_renderer::next_texunit()
    {
      return _texunit++;
    }

    /////////////////////////////////////////////////////////////////////////////
    program const& bezierobject_renderer::get_program() const
    {
      return *_program;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::modelviewmatrix(gpucast::math::matrix4f const& m)
    {
      _modelviewmatrix = m;

      _modelviewmatrixinverse = m;
      _modelviewmatrixinverse.invert();

      _normalmatrix = m.normalmatrix();
      _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::projectionmatrix(gpucast::math::matrix4f const& m)
    {
      _projectionmatrix = m;

      _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
      _modelviewprojectionmatrixinverse = gpucast::math::inverse(_modelviewprojectionmatrix);
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_nearfar(float near, float far)
    {
      _nearplane = near;
      _farplane = far;
      projectionmatrix(gpucast::math::frustum(-1.0f, 1.0f, -1.0f, 1.0f, _nearplane, _farplane));
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_background(gpucast::math::vec3f const& color)
    {
      _background = color;
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::add_search_path(std::string const& path)
    {
      _pathlist.insert(path);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::spheremap(std::string const& filepath)
    {
      _spheremap->load(filepath);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::diffusemap(std::string const& filepath)
    {
      _diffusemap->load(filepath);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::recompile()
    {
      _init_program();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::bind()
    {
      _program->begin();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::unbind()
    {
      _program->end();
      _texunit = 0;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::apply_uniforms()
    {
      // view parameters
      _program->set_uniform1f("nearplane", _nearplane);
      _program->set_uniform1f("farplane", _farplane);

      _program->set_uniform_matrix4fv("normalmatrix", 1, false, &_normalmatrix[0]);
      _program->set_uniform_matrix4fv("modelviewmatrix", 1, false, &_modelviewmatrix[0]);
      _program->set_uniform_matrix4fv("modelviewmatrixinverse", 1, false, &_modelviewmatrixinverse[0]);
      _program->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &_modelviewprojectionmatrix[0]);

      if (_spheremap) {
        _program->set_uniform1i("spheremapping", 1);
        _program->set_texture2d("spheremap", *_spheremap, next_texunit());
      } else {
        _program->set_uniform1i("spheremapping", 0);
      }
      
      if (_diffusemap) {
        _program->set_uniform1i("diffusemapping", 1);
        _program->set_texture2d("diffusemap", *_diffusemap, next_texunit());
      } else {
        _program->set_uniform1i("diffusemapping", 0);
      }

      if (_prefilter_texture) {
        _program->set_texture2d("prefilter_texture", *_prefilter_texture, next_texunit());
      }

    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_program()
    {
      gpucast::gl::resource_factory factory;

      _program = factory.create_program("resources/glsl/trimmed_surface/raycast_surface.glsl.vert", "resources/glsl/trimmed_surface/raycast_surface.glsl.frag");
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_prefilter(unsigned prefilter_resolution) 
    {
      _prefilter_texture = std::make_shared<gpucast::gl::texture2d>();

      gpucast::math::util::prefilter2d<gpucast::math::vec2d> pre_integrator(32, 0.5);

      
      std::vector<float> texture_data;

      auto distance_offset = std::sqrt(2) / prefilter_resolution;
      auto angle_offset = (2.0 * M_PI) / prefilter_resolution;

      for (unsigned d = 0; d != prefilter_resolution; ++d) {
        for (unsigned a = 0; a != prefilter_resolution; ++a) {

          auto angle = a * angle_offset;
          auto distance = -1.0 / std::sqrt(2) + distance_offset * d;
          auto alpha = pre_integrator(gpucast::math::vec2d(angle, distance));

          texture_data.push_back(float(alpha));
        }
      }

      _prefilter_texture->teximage(0, GL_R32F, prefilter_resolution, prefilter_resolution, 0, GL_RED, GL_FLOAT, (void*)(&texture_data[0]));
    }

  } // namespace gl
} // namespace gpucast 
