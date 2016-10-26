/********************************************************************************
*
* Copyright (C) 2014 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bezierobject_renderer.cpp
*  project    : gpucast::gl
*  description:
*
********************************************************************************/
#include "gpucast/gl/primitives/bezierobject_renderer.hpp"

#include <fstream>
#include <regex>

#include <gpucast/gl/error.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/util/resource_factory.hpp>

#include <gpucast/math/util/prefilter2d.hpp>

#include <gpucast/core/config.hpp>

#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>

#include <gpucast/core/singleton.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_kd.hpp>
#include <gpucast/core/trimdomain_serializer_double_binary.hpp>
#include <gpucast/core/trimdomain_serializer_loop_contour_list.hpp>

namespace gpucast {
  namespace gl {

    /////////////////////////////////////////////////////////////////////////////
    bezierobject_renderer::bezierobject_renderer() 
    {
      _pathlist.insert("");

      _init_raycasting_program();
      _init_pretesselation_program();
      _init_tesselation_program();

      _init_hullvertexmap();
      _init_prefilter();
      _init_transform_feedback();

      _counter = std::make_shared<atomicbuffer>(2 * sizeof(unsigned), GL_DYNAMIC_COPY);
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject_renderer::~bezierobject_renderer()
    {}

    /////////////////////////////////////////////////////////////////////////////
    int bezierobject_renderer::next_texunit()
    {
      return _texunit++;
    }

    /////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<program> const& bezierobject_renderer::get_raycasting_program() const
    {
      return _raycasting_program;
    }

    /////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<program> const& bezierobject_renderer::get_pretesselation_program() const
    {
      return _pretesselation_program;
    }

    /////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<program> const& bezierobject_renderer::get_tesselation_program() const
    {
      return _tesselation_program;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::current_viewmatrix(gpucast::math::matrix4f const& m)
    {
      _viewmatrix = m;
      _viewmatrixinverse = gpucast::math::inverse(_viewmatrix);

      // update other matrices
      _modelviewmatrix = _viewmatrix * _modelmatrix;
      _modelviewmatrixinverse = gpucast::math::inverse(_modelviewmatrix);

      _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
      _modelviewprojectionmatrixinverse = gpucast::math::inverse(_modelviewprojectionmatrix);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::current_modelmatrix(gpucast::math::matrix4f const& m)
    {
      _modelmatrix = m;
      _modelmatrixinverse = gpucast::math::inverse(_modelmatrix);

      // update other matrices
      _modelviewmatrix = _viewmatrix * _modelmatrix;
      _modelviewmatrixinverse = gpucast::math::inverse(_modelviewmatrix);

      _normalmatrix = _modelmatrix.normalmatrix();

      _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
      _modelviewprojectionmatrixinverse = gpucast::math::inverse(_modelviewprojectionmatrix);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::current_projectionmatrix(gpucast::math::matrix4f const& m)
    {
      _projectionmatrix = m;
      _projectionmatrixinverse = gpucast::math::inverse(_projectionmatrix);

      _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
      _modelviewprojectionmatrixinverse = gpucast::math::inverse(_modelviewprojectionmatrix);
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_nearfar(float near, float far)
    {
      _nearplane = near;
      _farplane = far;
      current_projectionmatrix(gpucast::math::frustum(-1.0f, 1.0f, -1.0f, 1.0f, _nearplane, _farplane));
    }

    void bezierobject_renderer::set_resolution(unsigned width, unsigned height)
    {
      _resolution = gpucast::math::vec2i(width, height);
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
      _spheremap = std::make_shared<texture2d>();
      _spheremap->load(filepath);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::diffusemap(std::string const& filepath)
    {
      _diffusemap = std::make_shared<texture2d>();
      _diffusemap->load(filepath);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::recompile()
    {
      _init_raycasting_program();
      _init_pretesselation_program();
      _init_tesselation_program();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::begin_program(std::shared_ptr<program> const& p)
    {
      p->begin();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::end_program(std::shared_ptr<program> const& p)
    {
      p->end();
      _texunit = 0;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::apply_uniforms(std::shared_ptr<program> const& p)
    {
      // view parameters
      p->set_uniform1f("gpucast_clip_near", _nearplane);
      p->set_uniform1f("gpucast_clip_far", _farplane);

      p->set_uniform2i("gpucast_resolution", _resolution[0], _resolution[1]);
      if (p == _tesselation_program || p == _pretesselation_program) {
        p->set_shaderstoragebuffer("gpucast_hullvertexmap_ssbo", *_hullvertexmap, GPUCAST_HULLVERTEXMAP_SSBO_BINDING);
      }

      if (_enable_count)
      {
        p->set_uniform1i("gpucast_enable_counting", _enable_count);
        _counter->bind_buffer_base(bezierobject_renderer::GPUCAST_ATOMIC_COUNTER_BINDING);
      }

      // camera block
      p->set_uniform_matrix4fv("gpucast_projection_matrix", 1, false, &_projectionmatrix[0]);
      p->set_uniform_matrix4fv("gpucast_projection_inverse_matrix", 1, false, &_projectionmatrixinverse[0]);

      p->set_uniform_matrix4fv("gpucast_model_matrix", 1, false, &_modelmatrix[0]);
      p->set_uniform_matrix4fv("gpucast_model_inverse_matrix", 1, false, &_modelmatrixinverse[0]);

      p->set_uniform_matrix4fv("gpucast_view_matrix", 1, false, &_viewmatrix[0]);
      p->set_uniform_matrix4fv("gpucast_view_inverse_matrix", 1, false, &_viewmatrixinverse[0]);

      p->set_uniform_matrix4fv("gpucast_normal_matrix", 1, false, &_normalmatrix[0]);
      p->set_uniform_matrix4fv("gpucast_model_view_matrix", 1, false, &_modelviewmatrix[0]);
      p->set_uniform_matrix4fv("gpucast_model_view_inverse_matrix", 1, false, &_modelviewmatrixinverse[0]);

      p->set_uniform_matrix4fv("gpucast_model_view_projection_matrix", 1, false, &_modelviewprojectionmatrix[0]);
      p->set_uniform_matrix4fv("gpucast_model_view_projection_inverse_matrix", 1, false, &_modelviewprojectionmatrixinverse[0]);

      if (_spheremap) {
        p->set_uniform1i("spheremapping", 1);
        p->set_texture2d("spheremap", *_spheremap, next_texunit());
      } else {
        p->set_uniform1i("spheremapping", 0);
      }
      
      if (_diffusemap) {
        p->set_uniform1i("diffusemapping", 1);
        p->set_texture2d("diffusemap", *_diffusemap, next_texunit());
      } else {
        p->set_uniform1i("diffusemapping", 0);
      }
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::enable_counting(bool b)
    {
      _enable_count = b;
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject_renderer::enable_counting() const
    {
      return _enable_count;
    }

    /////////////////////////////////////////////////////////////////////////////
    unsigned bezierobject_renderer::get_triangle_count() const
    {
      _counter->bind();
      unsigned* mapped_mem_read = (unsigned*)_counter->map_range(0, sizeof(unsigned), GL_MAP_READ_BIT);
      unsigned result = mapped_mem_read[0];
      _counter->unmap();
      _counter->unbind();
      return result;
    }

    /////////////////////////////////////////////////////////////////////////////
    unsigned bezierobject_renderer::get_fragment_count() const
    {
      _counter->bind();
      unsigned* mapped_mem_read = (unsigned*)_counter->map_range(0, sizeof(unsigned), GL_MAP_READ_BIT);
      unsigned result = mapped_mem_read[1];
      _counter->unmap();
      _counter->unbind();
      return result;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::reset_count() const
    {
      // initialize buffer with 0
      _counter->bind();
      unsigned* mapped_mem_write = (unsigned*)_counter->map_range(0, 2 * sizeof(unsigned), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      mapped_mem_write[0] = 0; // triangle count
      mapped_mem_write[1] = 0; // fragment count
      _counter->unmap();
      _counter->unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_raycasting_program()
    {
      BOOST_LOG_TRIVIAL(info) << "_init_raycasting_program()" << std::endl;

      gpucast::gl::resource_factory factory;

      _raycasting_program = factory.create_program({ 
        { vertex_stage, "resources/glsl/trimmed_surface/raycast_surface.glsl.vert" },
        { fragment_stage, "resources/glsl/trimmed_surface/raycast_surface.glsl.frag" } 
      });

    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_pretesselation_program()
    {
      BOOST_LOG_TRIVIAL(info) << "_init_pretesselation_program()" << std::endl;

      gpucast::gl::resource_factory factory;

      _pretesselation_program = factory.create_program({ 
        { vertex_stage, "resources/glsl/trimmed_surface/pretesselation.vert.glsl"},
        { tesselation_control_stage, "resources/glsl/trimmed_surface/pretesselation.tctrl.glsl"},                                                         
        { tesselation_evaluation_stage, "resources/glsl/trimmed_surface/pretesselation.teval.glsl" },
        { geometry_stage, "resources/glsl/trimmed_surface/pretesselation.geom.glsl" } 
      });

      const char *varyings[] =
      {
        "transform_position",
        "transform_index",
        "transform_tesscoord",
        "transform_final_tesselation"
      };

      glTransformFeedbackVaryings(_pretesselation_program->id(), 4, (char**)&varyings, GL_INTERLEAVED_ATTRIBS);

      _pretesselation_program->link();
      BOOST_LOG_TRIVIAL(info) << _pretesselation_program->log() << std::endl;;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_tesselation_program()
    {
      gpucast::gl::resource_factory factory;
      
      _tesselation_program = factory.create_program({ 
        { vertex_stage, "resources/glsl/trimmed_surface/tesselation.vert.glsl" },
        { tesselation_control_stage, "resources/glsl/trimmed_surface/tesselation.tctrl.glsl" },
        { tesselation_evaluation_stage, "resources/glsl/trimmed_surface/tesselation.teval.glsl" },
        { geometry_stage, "resources/glsl/trimmed_surface/tesselation.geom.glsl" },
        { fragment_stage, "resources/glsl/trimmed_surface/tesselation.frag.glsl" } 
      });
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_hullvertexmap() 
    {
      _hullvertexmap = std::make_shared<shaderstoragebuffer>();

      hullvertexmap hvm;

      _hullvertexmap->update(hvm.data.begin(), hvm.data.end());
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

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_transform_feedback()
    {
      BOOST_LOG_TRIVIAL(info) << "bezierobject_renderer::_init_transform_feedback(): Reserved memory = " << MAX_XFB_BUFFER_SIZE_IN_BYTES/(1024*1024) << " MBytes." << std::endl;

      auto tfbuffer = singleton<transform_feedback_buffer>::instance();

      if (tfbuffer->feedback == 0)
      {
        // initialize objects
        tfbuffer->feedback = std::make_shared<gpucast::gl::transform_feedback>();
        tfbuffer->vertex_array_object = std::make_shared<gpucast::gl::vertexarrayobject>();
        tfbuffer->buffer = std::make_shared<gpucast::gl::arraybuffer>(MAX_XFB_BUFFER_SIZE_IN_BYTES, GL_STATIC_DRAW);

        // bind array buffer as target to transform feedback
        tfbuffer->feedback->bind();
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfbuffer->buffer->id());
        tfbuffer->feedback->unbind();

        // setup transform feedback vertex array
        int stride = sizeof(math::vec3f) + sizeof(unsigned) + sizeof(math::vec2f) + sizeof(math::vec3f);
        tfbuffer->vertex_array_object->bind();
        {
          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 0, 3, GL_FLOAT, false, stride, 0);
          tfbuffer->vertex_array_object->enable_attrib(0);

          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 1, 1, GL_UNSIGNED_INT, false, stride, sizeof(gpucast::math::vec3f));
          tfbuffer->vertex_array_object->enable_attrib(1);

          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 2, 2, GL_FLOAT, false, stride, sizeof(gpucast::math::vec3f) + sizeof(unsigned));
          tfbuffer->vertex_array_object->enable_attrib(2);

          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 3, 3, GL_FLOAT, false, stride, sizeof(gpucast::math::vec3f) + sizeof(unsigned) + sizeof(gpucast::math::vec2f));
          tfbuffer->vertex_array_object->enable_attrib(3);
        }
        tfbuffer->vertex_array_object->unbind();
      }
    }

  } // namespace gl
} // namespace gpucast 

