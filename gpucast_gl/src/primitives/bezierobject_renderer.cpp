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
      // configure program generation and initialization
      _pathlist.insert("");

      _program_factory.add_substitution("GPUCAST_MAX_FEEDBACK_BUFFER_INDICES_INPUT", std::to_string(MAX_FEEDBACK_BUFFER_INDICES));
      _program_factory.add_substitution("GPUCAST_SECOND_PASS_TRIANGLE_TESSELATION_INPUT", std::to_string(_enable_triangular_tesselation));
      _program_factory.add_substitution("GPUCAST_WRITE_DEBUG_COUNTER_INPUT", std::to_string(_enable_count));
      _program_factory.add_substitution("GPUCAST_ANTI_ALIASING_MODE_INPUT", std::to_string(_antialiasing));

      _program_factory.add_substitution("GPUCAST_HULLVERTEXMAP_SSBO_BINDING_INPUT", std::to_string(GPUCAST_HULLVERTEXMAP_SSBO_BINDING));
      _program_factory.add_substitution("GPUCAST_ATTRIBUTE_SSBO_BINDING_INPUT", std::to_string(GPUCAST_ATTRIBUTE_SSBO_BINDING));
      _program_factory.add_substitution("GPUCAST_ATOMIC_COUNTER_BINDING_INPUT", std::to_string(GPUCAST_ATOMIC_COUNTER_BINDING));
      _program_factory.add_substitution("GPUCAST_FEEDBACK_BUFFER_BINDING_INPUT", std::to_string(GPUCAST_FEEDBACK_BUFFER_BINDING));
      _program_factory.add_substitution("GPUCAST_HOLE_FILLING_INPUT", std::to_string(_enable_holefilling));

      _program_factory.add_substitution("GPUCAST_ABUFFER_ATOMIC_BUFFER_BINDING_INPUT", std::to_string(GPUCAST_ABUFFER_ATOMIC_BUFFER_BINDING));
      _program_factory.add_substitution("GPUCAST_ABUFFER_FRAGMENT_LIST_BUFFER_BINDING_INPUT", std::to_string(GPUCAST_ABUFFER_FRAGMENT_LIST_BUFFER_BINDING));
      _program_factory.add_substitution("GPUCAST_ABUFFER_FRAGMENT_DATA_BUFFER_BINDING_INPUT", std::to_string(GPUCAST_ABUFFER_FRAGMENT_DATA_BUFFER_BINDING));

      // build programs
      recompile();

      // init auxilliary structures
      _init_hullvertexmap();
      _init_prefilter();
      _init_transform_feedback();

      // init debug data structures
      _feedbackbuffer = std::make_shared<shaderstoragebuffer>(MAX_FEEDBACK_BUFFER_INDICES * sizeof(unsigned), GL_DYNAMIC_COPY);
      _counter = std::make_shared<atomicbuffer>(sizeof(debug_counter), GL_DYNAMIC_COPY);

      // create offscreen targets
      _fbo = std::make_shared<framebufferobject>();
      _fbo_multisample = std::make_shared<framebufferobject>();
      _gbuffer = std::make_shared<framebufferobject>();

      _gbuffer_colorattachment = std::make_shared<texture2d>();
      _gbuffer_depthattachment = std::make_shared<texture2d>();

      _colorattachment = std::make_shared<texture2d>();
      _depthattachment = std::make_shared<texture2d>();

      _colorattachment_multisample = std::make_shared<renderbuffer>();
      _depthattachment_multisample = std::make_shared<renderbuffer>();

      _camera_ubo = std::make_shared<uniformbuffer>();
      _camera_ubo->bufferdata(sizeof(matrix_uniform_buffer_layout), 0, GL_DYNAMIC_DRAW);

      // fullscreen pass geometry
      _fullscreen_quad = std::make_shared<gpucast::gl::plane>(0, -1, 1);
      //_fullscreen_quad->size(1.0, 1.0);

      _nearest_sampler = std::make_shared<sampler>();
      _nearest_sampler->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
      _nearest_sampler->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
      _nearest_sampler->parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      _nearest_sampler->parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

      _linear_sampler = std::make_shared<sampler>();
      _linear_sampler->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
      _linear_sampler->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
      _linear_sampler->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      _linear_sampler->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      const unsigned abuffer_list_size = _abuffer_max_fragment * 2 * sizeof(unsigned);
      const unsigned abuffer_data_size = _abuffer_max_fragment * sizeof(gpucast::math::vec4u);

      BOOST_LOG_TRIVIAL(info) << "bezierobject_renderer::bezierobject_renderer() : Allocating " << (abuffer_list_size + abuffer_data_size) / (1024 * 1024) << "MB for A-Buffer storage.";

      _abuffer_fragment_list = std::make_shared<shaderstoragebuffer>(abuffer_list_size, GL_DYNAMIC_COPY);
      _abuffer_fragment_data = std::make_shared<shaderstoragebuffer>(abuffer_data_size, GL_DYNAMIC_COPY);

      _abuffer_atomic_buffer = std::make_shared<atomicbuffer>(2*sizeof(unsigned), GL_DYNAMIC_COPY);
      _abuffer_atomic_buffer->clear_data(GL_RG32UI, GL_RGB, GL_UNSIGNED_INT, 0);
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject_renderer::~bezierobject_renderer()
    {}

    ////////////////////////// ///////////////////////////////////////////////////
    int bezierobject_renderer::next_texunit()
    {
      return _texunit++;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_custom_transform_feedback_size(std::size_t bytes) {
      auto tfbuffer = singleton<transform_feedback_buffer>::instance();
      tfbuffer->feedback.reset();
      _init_transform_feedback();
    }

    void bezierobject_renderer::set_abuffer_size(std::size_t max_fragments)
    {
      _abuffer_max_fragment = max_fragments;

      const unsigned abuffer_list_size = max_fragments * 2 * sizeof(unsigned);
      const unsigned abuffer_data_size = max_fragments * sizeof(gpucast::math::vec4u);

      BOOST_LOG_TRIVIAL(info) << "bezierobject_renderer::bezierobject_renderer() : Allocating " << abuffer_list_size + abuffer_data_size << "bytes for A-Buffer storage.";

      _abuffer_fragment_list = std::make_shared<shaderstoragebuffer>(abuffer_list_size, GL_DYNAMIC_DRAW);
      _abuffer_fragment_data = std::make_shared<shaderstoragebuffer>(abuffer_data_size, GL_DYNAMIC_DRAW);
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
    void bezierobject_renderer::view_setup(gpucast::math::matrix4f const& view, gpucast::math::matrix4f const& model, gpucast::math::matrix4f const& projection)
    {
      _camera_ubo_data.gpucast_view_matrix = view;
      _camera_ubo_data.gpucast_model_matrix = model;
      _camera_ubo_data.gpucast_projection_matrix = projection;

      _camera_ubo_data.gpucast_view_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_view_matrix);

      _camera_ubo_data.gpucast_normal_matrix = _camera_ubo_data.gpucast_model_matrix.normalmatrix();

      _camera_ubo_data.gpucast_model_view_matrix = _camera_ubo_data.gpucast_view_matrix * _camera_ubo_data.gpucast_model_matrix;
      _camera_ubo_data.gpucast_model_view_projection_matrix = _camera_ubo_data.gpucast_projection_matrix  * _camera_ubo_data.gpucast_model_view_matrix;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::current_viewmatrix(gpucast::math::matrix4f const& m)
    {
      _camera_ubo_data.gpucast_view_matrix = m;
      _camera_ubo_data.gpucast_view_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_view_matrix);

      // update other matrices
      _camera_ubo_data.gpucast_model_view_matrix = _camera_ubo_data.gpucast_view_matrix * _camera_ubo_data.gpucast_model_matrix;
      _camera_ubo_data.gpucast_model_view_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_model_view_matrix);

      _camera_ubo_data.gpucast_model_view_projection_matrix = _camera_ubo_data.gpucast_projection_matrix * _camera_ubo_data.gpucast_model_view_matrix;
      _camera_ubo_data.gpucast_model_view_projection_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_model_view_projection_matrix);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::current_modelmatrix(gpucast::math::matrix4f const& m)
    {
      _camera_ubo_data.gpucast_model_matrix = m;
      _camera_ubo_data.gpucast_model_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_model_matrix);

      // update other matrices
      _camera_ubo_data.gpucast_model_view_matrix = _camera_ubo_data.gpucast_view_matrix * _camera_ubo_data.gpucast_model_matrix;
      _camera_ubo_data.gpucast_model_view_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_model_view_matrix);

      _camera_ubo_data.gpucast_normal_matrix = _camera_ubo_data.gpucast_model_matrix.normalmatrix();

      _camera_ubo_data.gpucast_model_view_projection_matrix = _camera_ubo_data.gpucast_projection_matrix * _camera_ubo_data.gpucast_model_view_matrix;
      _camera_ubo_data.gpucast_model_view_projection_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_model_view_projection_matrix);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::current_projectionmatrix(gpucast::math::matrix4f const& m)
    {
      _camera_ubo_data.gpucast_projection_matrix = m;
      _camera_ubo_data.gpucast_projection_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_projection_matrix);

      _camera_ubo_data.gpucast_model_view_projection_matrix = _camera_ubo_data.gpucast_projection_matrix * _camera_ubo_data.gpucast_model_view_matrix;
      _camera_ubo_data.gpucast_model_view_projection_inverse_matrix = gpucast::math::inverse(_camera_ubo_data.gpucast_model_view_projection_matrix);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::attach_custom_textures(std::shared_ptr<texture2d> const& color_texture,
      std::shared_ptr<texture2d> const& depth_texture)
    {
      _colorattachment = color_texture;
      _depthattachment = depth_texture;

      create_fbo();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_nearfar(float near, float far)
    {
      _camera_ubo_data.gpucast_clip_near = near;
      _camera_ubo_data.gpucast_clip_far = far;
      current_projectionmatrix(gpucast::math::frustum(-1.0f, 1.0f, -1.0f, 1.0f, _camera_ubo_data.gpucast_clip_near, _camera_ubo_data.gpucast_clip_far));
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_resolution(unsigned width, unsigned height)
    {
      if (width == _camera_ubo_data.gpucast_resolution[0] && height == _camera_ubo_data.gpucast_resolution[1]) {
        return;
      }
      else {

        // recreate offscreen targets with new textures
        _camera_ubo_data.gpucast_resolution = gpucast::math::vec2i(width, height);

        // resize fbo textures
        _colorattachment->teximage(0, GL_RGBA32F, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]), 0, GL_RGBA, GL_FLOAT, 0);
        _depthattachment->teximage(0, GL_DEPTH32F_STENCIL8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

        _gbuffer_colorattachment->teximage(0, GL_RGBA32F, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]), 0, GL_RGBA, GL_FLOAT, 0);
        _gbuffer_depthattachment->teximage(0, GL_DEPTH32F_STENCIL8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

        allocate_multisample_textures();

        // update FBOs 
        create_fbo();
        create_gbuffer();
      }
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::allocate_multisample_textures()
    {
      // resize multisample textures
      switch (_antialiasing) {
      case gpucast::gl::bezierobject::csaa4:
      {
        _colorattachment_multisample->set(4, GL_RGBA8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]));
        _depthattachment_multisample->set(4, GL_DEPTH32F_STENCIL8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]));
        break;
      }
      case gpucast::gl::bezierobject::csaa8:
      {
        _colorattachment_multisample->set(8, GL_RGBA8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]));
        _depthattachment_multisample->set(8, GL_DEPTH32F_STENCIL8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]));
        break;
      }
      case gpucast::gl::bezierobject::csaa16:
      {
        _colorattachment_multisample->set(16, GL_RGBA8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]));
        _depthattachment_multisample->set(16, GL_DEPTH32F_STENCIL8, GLsizei(_camera_ubo_data.gpucast_resolution[0]), GLsizei(_camera_ubo_data.gpucast_resolution[1]));
        break;
      }

      };

      create_multisample_fbo();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::create_fbo()
    {
      _fbo->attach_texture(*_colorattachment, GL_COLOR_ATTACHMENT0_EXT);
      _fbo->attach_texture(*_depthattachment, GL_DEPTH_STENCIL_ATTACHMENT);

      _fbo->bind();
      _fbo->status();
      _fbo->unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::create_multisample_fbo()
    {
      _fbo_multisample->attach_renderbuffer(*_colorattachment_multisample, GL_COLOR_ATTACHMENT0_EXT);
      _fbo_multisample->attach_renderbuffer(*_depthattachment_multisample, GL_DEPTH_STENCIL_ATTACHMENT);

      _fbo_multisample->bind();
      _fbo_multisample->status();
      _fbo_multisample->unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::create_gbuffer()
    {
      _gbuffer->attach_texture(*_gbuffer_colorattachment, GL_COLOR_ATTACHMENT0_EXT);
      _gbuffer->attach_texture(*_gbuffer_depthattachment, GL_DEPTH_STENCIL_ATTACHMENT);

      _gbuffer->bind();
      _gbuffer->status();
      _gbuffer->unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::begin_draw() 
    {
      // first clear abuffer
      _abuffer_clear();

      if (_antialiasing == gpucast::gl::bezierobject::csaa4 || _antialiasing == gpucast::gl::bezierobject::csaa8 || _antialiasing == gpucast::gl::bezierobject::csaa16) {
        _fbo_multisample->bind(); 
      } else {
        _gbuffer->bind();
      }

      // backup GL state
      glGetIntegerv(GL_POLYGON_MODE, &_glstate_backup._polygonmode);
      _glstate_backup._conservative_rasterization_enabled = glIsEnabled(GL_CONSERVATIVE_RASTERIZATION_NV);
      if (!_glstate_backup._conservative_rasterization_enabled && _conservative_rasterization) {
        glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
      }

      glClearColor(_background[0], _background[1], _background[2], 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::end_draw() 
    {
      // blit FBO if MSAA is enabled 
      if (_antialiasing == gpucast::gl::bezierobject::csaa4 || _antialiasing == gpucast::gl::bezierobject::csaa8 || _antialiasing == gpucast::gl::bezierobject::csaa16)
      {
        _fbo_multisample->unbind();

        _gbuffer->bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        _gbuffer->unbind();

        glBlitNamedFramebuffer(_fbo_multisample->id(), _gbuffer->id(),
          0, 0, _camera_ubo_data.gpucast_resolution[0], _camera_ubo_data.gpucast_resolution[1], 0, 0, _camera_ubo_data.gpucast_resolution[0], _camera_ubo_data.gpucast_resolution[1],
          GL_COLOR_BUFFER_BIT, GL_NEAREST);
      }
      else {
        _gbuffer->unbind();
      }

#if 1

      // restore GL state
      if (!_glstate_backup._conservative_rasterization_enabled) {
        glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
      } 
      glPolygonMode(GL_FRONT_AND_BACK, _glstate_backup._polygonmode);
      
      glEnable(GL_DEPTH_TEST);
      glClearDepth(1.0f);

      // resolve into FBO
      _fbo->bind();
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

      begin_program(_resolve_program);
      {
        _resolve_program->set_uniformbuffer("gpucast_matrix_uniforms", *_camera_ubo, GPUCAST_CAMERA_UBO_BINDINGPOINT);

        _resolve_program->set_texture2d("gpucast_gbuffer_color", *_gbuffer_colorattachment, 1);
        _nearest_sampler->bind(1);

        _resolve_program->set_texture2d("gpucast_gbuffer_depth", *_gbuffer_depthattachment, 2);
        _nearest_sampler->bind(2);

        _fullscreen_quad->draw();
      }
      end_program(_resolve_program);

      _fbo->unbind();
#endif
    }

    /////////////////////////////////////////////////////////////////////////////
    gpucast::math::vec2i const& bezierobject_renderer::get_resolution() const
    {
      return _camera_ubo_data.gpucast_resolution;
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
    void bezierobject_renderer::spheremapping(bool b)
    {
      _enable_spheremap = b;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::spheremap(std::string const& filepath)
    {
      _spheremap = std::make_shared<texture2d>();
      _spheremap->load(filepath);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::diffusemapping(bool b)
    {
      _enable_diffusemap = b;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::diffusemap(std::string const& filepath)
    {
      _diffusemap = std::make_shared<texture2d>();
      _diffusemap->load(filepath);
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject_renderer::inside_frustum(bezierobject const& bo) const
    {
      // get objects bounding box in world space
      auto bbox_min = gpucast::math::point3f(bo.object().bbox().min);
      auto bbox_max = gpucast::math::point3f(bo.object().bbox().max);

      bbox_min.weight(1.0f);
      bbox_max.weight(1.0f);

      gpucast::math::axis_aligned_boundingbox<gpucast::math::point3f> bbox(bbox_min, bbox_max);

      // compute planes
      std::array<gpucast::math::vec4f, 6> planes;
      auto projection_view = _camera_ubo_data.gpucast_projection_matrix * _camera_ubo_data.gpucast_view_matrix;// *_camera_ubo_data.gpucast_model_matrix;

      //left plane
      planes[0] = gpucast::math::vec4f(projection_view[3] + projection_view[0],
        projection_view[7] + projection_view[4],
        projection_view[11] + projection_view[8],
        projection_view[15] + projection_view[12]);

      //right plane
      planes[1] = gpucast::math::vec4f(projection_view[3] - projection_view[0],
        projection_view[7] - projection_view[4],
        projection_view[11] - projection_view[8],
        projection_view[15] - projection_view[12]);

      //bottom plane
      planes[2] = gpucast::math::vec4f(projection_view[3] + projection_view[1],
        projection_view[7] + projection_view[5],
        projection_view[11] + projection_view[9],
        projection_view[15] + projection_view[13]);

      //top plane
      planes[3] = gpucast::math::vec4f(projection_view[3] - projection_view[1],
        projection_view[7] - projection_view[5],
        projection_view[11] - projection_view[9],
        projection_view[15] - projection_view[13]);

      //near plane
      planes[4] = gpucast::math::vec4f(projection_view[3] + projection_view[2],
        projection_view[7] + projection_view[6],
        projection_view[11] + projection_view[10],
        projection_view[15] + projection_view[14]);

      //far plane
      planes[5] = gpucast::math::vec4f(projection_view[3] - projection_view[2],
        projection_view[7] - projection_view[6],
        projection_view[11] - projection_view[10],
        projection_view[15] - projection_view[14]);

      auto outside = [](gpucast::math::vec4f const& plane, gpucast::math::vec3f const& point) {
        return plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3] < 0;
      };

      for (unsigned i(0); i < 6; ++i) 
      {
        auto p(bbox.min);
        if (planes[i][0] >= 0)
          p[0] = bbox.max[0];
        if (planes[i][1] >= 0)
          p[1] = bbox.max[1];
        if (planes[i][2] >= 0)
          p[2] = bbox.max[2];

        // is the positive vertex outside?
        if (outside(planes[i], p)) {
          return false;
        }
      }

      return true;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::recompile()
    {
      _init_raycasting_program();
      _init_pretesselation_program();
      _init_tesselation_program();
      _init_resolve_program();
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
      if (p == _tesselation_program || p == _pretesselation_program) {
        p->set_shaderstoragebuffer("gpucast_hullvertexmap_ssbo", *_hullvertexmap, GPUCAST_HULLVERTEXMAP_SSBO_BINDING);
      }

      if (_enable_count && p == _pretesselation_program)
      {
        _counter->bind_buffer_base(bezierobject_renderer::GPUCAST_ATOMIC_COUNTER_BINDING);
        p->set_shaderstoragebuffer("gpucast_feedback_buffer", *_feedbackbuffer, GPUCAST_FEEDBACK_BUFFER_BINDING);
      }

      // update and bind UBO
      auto hostmem = _camera_ubo->map(GL_WRITE_ONLY);
      std::memcpy(hostmem, &_camera_ubo_data, sizeof(matrix_uniform_buffer_layout));
      _camera_ubo->unmap();
      p->set_uniformbuffer("gpucast_matrix_uniforms", *_camera_ubo, GPUCAST_CAMERA_UBO_BINDINGPOINT);

      if (_antialiasing != gpucast::gl::bezierobject::csaa4 && _antialiasing != gpucast::gl::bezierobject::csaa8 && _antialiasing != gpucast::gl::bezierobject::csaa16) {
        auto depthtex_unit = next_texunit();
        p->set_texture2d("gpucast_depth_buffer", *_depthattachment, depthtex_unit);
        _nearest_sampler->bind(depthtex_unit);
      }
      else {
        //BOOST_LOG_TRIVIAL(info) << "Warning: Depth buffer cannot be bound in MSAA mode.";
      }

      if (p == _tesselation_program) {
        _abuffer_atomic_buffer->bind_buffer_base(bezierobject_renderer::GPUCAST_ABUFFER_ATOMIC_BUFFER_BINDING);
        p->set_shaderstoragebuffer("gpucast_abuffer_list", *_abuffer_fragment_list, GPUCAST_ABUFFER_FRAGMENT_LIST_BUFFER_BINDING);
        p->set_shaderstoragebuffer("gpucast_abuffer_data", *_abuffer_fragment_data, GPUCAST_ABUFFER_FRAGMENT_DATA_BUFFER_BINDING);
      }

      if (_spheremap) {
        auto unit = next_texunit();
        p->set_uniform1i("spheremapping", int(_enable_spheremap));
        p->set_texture2d("spheremap", *_spheremap, unit);
        _linear_sampler->bind(unit);
      } else {
        p->set_uniform1i("spheremapping", 0);
      }
      
      if (_diffusemap) {
        p->set_uniform1i("diffusemapping", int(_enable_diffusemap));
        p->set_texture2d("diffusemap", *_diffusemap, next_texunit());
      } else {
        p->set_uniform1i("diffusemapping", 0);
      }

      auto prefilter_unit = next_texunit();
      p->set_texture2d("gpucast_prefilter", *_prefilter_texture, prefilter_unit);
      _linear_sampler->bind(prefilter_unit);
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::antialiasing(bezierobject::anti_aliasing_mode m)
    {
      _antialiasing = m;
      _program_factory.add_substitution("GPUCAST_ANTI_ALIASING_MODE_INPUT", std::to_string(_antialiasing));
      recompile();
      allocate_multisample_textures();
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::anti_aliasing_mode bezierobject_renderer::antialiasing() const
    {
      return _antialiasing;
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::enable_conservative_rasterization(bool b)
    {
      _conservative_rasterization = b;
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject_renderer::enable_conservative_rasterization() const
    {
      return _conservative_rasterization;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::enable_holefilling(bool b)
    {
      _enable_holefilling = b;
      _program_factory.add_substitution("GPUCAST_HOLE_FILLING_INPUT", std::to_string(_enable_holefilling));
      recompile();
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject_renderer::enable_holefilling() const
    {
      return _enable_holefilling;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::enable_counting(bool b)
    {
      _enable_count = b;
      _program_factory.add_substitution("GPUCAST_WRITE_DEBUG_COUNTER_INPUT", std::to_string(_enable_count));
      recompile();
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject_renderer::enable_counting() const
    {
      return _enable_count;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::enable_triangular_tesselation(bool b)
    {
      _enable_triangular_tesselation = b;
      _program_factory.add_substitution("GPUCAST_SECOND_PASS_TRIANGLE_TESSELATION_INPUT", std::to_string(_enable_triangular_tesselation));
      recompile();
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject_renderer::enable_triangular_tesselation() const
    {
      return _enable_triangular_tesselation;
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject_renderer::debug_counter bezierobject_renderer::get_debug_count() const
    {
      _counter->bind();
      debug_counter result;
      unsigned* mapped_mem_read = (unsigned*)_counter->map_range(0, sizeof(debug_counter), GL_MAP_READ_BIT);
      std::memcpy(&result, mapped_mem_read, sizeof(debug_counter));
      _counter->unmap();

      // initialize buffer with 0
      unsigned* mapped_mem_write = (unsigned*)_counter->map_range(0, sizeof(debug_counter), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      std::fill_n(mapped_mem_write, sizeof(debug_counter) / sizeof(unsigned), 0);
      _counter->unmap();
      _counter->unbind();

      return result;
    }

    /////////////////////////////////////////////////////////////////////////////
    std::vector<unsigned> bezierobject_renderer::get_fragment_estimate() const
    {
      std::vector<unsigned> feedback(MAX_FEEDBACK_BUFFER_INDICES);
      _feedbackbuffer->bind();
      unsigned* mapped_mem_read = (unsigned*)_feedbackbuffer->map_range(0, sizeof(unsigned), GL_MAP_READ_BIT);
      if (mapped_mem_read) {
        for (auto i = 0; i != MAX_FEEDBACK_BUFFER_INDICES; ++i) {
          feedback[i] = mapped_mem_read[i];
        }
      }
      _feedbackbuffer->unmap();
      _feedbackbuffer->unbind();

      // clear feedback
      std::vector<unsigned> feedback_zeroed(MAX_FEEDBACK_BUFFER_INDICES, 0);
      _feedbackbuffer->update(feedback_zeroed.begin(), feedback_zeroed.end());

      return feedback;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_abuffer_clear()
    {
      // readback
#if 0
      _abuffer_atomic_buffer->bind();
      unsigned* ctr_read = reinterpret_cast<unsigned*>(_abuffer_atomic_buffer->map_range(0, sizeof(unsigned), GL_MAP_READ_BIT));

      if (ctr_read) {
        unsigned fragcount;
        std::memcpy(&fragcount, ctr_read, sizeof(unsigned));
        BOOST_LOG_TRIVIAL(info) << "transparent frags : " << fragcount;
      } else {
        BOOST_LOG_TRIVIAL(info) << "Unable to map buffer";
      }
      _abuffer_atomic_buffer->unmap();
      _abuffer_atomic_buffer->unbind();
#endif

#if 1
      _abuffer_atomic_buffer->bind();
      unsigned* ctr_clear = reinterpret_cast<unsigned*>(_abuffer_atomic_buffer->map_range(0, sizeof(unsigned), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT));
      if (ctr_clear) {
        *ctr_clear = 0;
      } else {
        BOOST_LOG_TRIVIAL(info) << "Unable to map buffer";
      }
      _abuffer_atomic_buffer->unmap();
      _abuffer_atomic_buffer->unbind();
#endif

#if 1
      const unsigned abuffer_list_size = _abuffer_max_fragment * 2 * sizeof(unsigned);
      const unsigned abuffer_data_size = _abuffer_max_fragment * sizeof(gpucast::math::vec4u);

      _abuffer_fragment_list->clear_subdata(GL_RG32UI, 0u, abuffer_list_size, GL_RGB, GL_UNSIGNED_INT, 0);
      _abuffer_fragment_data->clear_subdata(GL_RG32UI, 0u, abuffer_data_size, GL_RGB, GL_UNSIGNED_INT, 0);
#endif
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_raycasting_program()
    {
      _raycasting_program = _program_factory.create_program({
        { vertex_stage, "resources/glsl/trimmed_surface/raycast_surface.glsl.vert" },
        { fragment_stage, "resources/glsl/trimmed_surface/raycast_surface.glsl.frag" } 
      });

    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_pretesselation_program()
    {
      _pretesselation_program = _program_factory.create_program({
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

      glTransformFeedbackVaryings(_pretesselation_program->id(), sizeof(varyings)/sizeof(char*), (char**)&varyings, GL_INTERLEAVED_ATTRIBS);

      _pretesselation_program->link();
      BOOST_LOG_TRIVIAL(info) << _pretesselation_program->log() << std::endl;;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_tesselation_program()
    {
      _tesselation_program = _program_factory.create_program({
        { vertex_stage, "resources/glsl/trimmed_surface/tesselation.vert.glsl" },
        { tesselation_control_stage, "resources/glsl/trimmed_surface/tesselation.tctrl.glsl" },
        { tesselation_evaluation_stage, "resources/glsl/trimmed_surface/tesselation.teval.glsl" },
        { geometry_stage, "resources/glsl/trimmed_surface/tesselation.geom.glsl" },
        { fragment_stage, "resources/glsl/trimmed_surface/tesselation.frag.glsl" } 
      });
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_resolve_program()
    {
      _resolve_program = _program_factory.create_program({
        { vertex_stage, "resources/glsl/trimmed_surface/abuffer_resolve.vert.glsl" },
        { fragment_stage, "resources/glsl/trimmed_surface/abuffer_resolve.frag.glsl" }
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
      BOOST_LOG_TRIVIAL(info) << "bezierobject_renderer::_init_transform_feedback(): Reserved memory = " << _xfb_budget_bytes /(1024*1024) << " MBytes." << std::endl;

      auto tfbuffer = singleton<transform_feedback_buffer>::instance();

      if (tfbuffer->feedback == 0)
      {
        // initialize objects
        tfbuffer->feedback = std::make_shared<gpucast::gl::transform_feedback>();
        tfbuffer->vertex_array_object = std::make_shared<gpucast::gl::vertexarrayobject>();
        tfbuffer->buffer = std::make_shared<gpucast::gl::arraybuffer>(_xfb_budget_bytes, GL_DYNAMIC_DRAW);

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

