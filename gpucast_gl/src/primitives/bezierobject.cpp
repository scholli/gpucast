/********************************************************************************
*
* Copyright (C) 2014 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bezierobject.cpp
*  project    : gpucast::gl
*  description:
*
********************************************************************************/
#include "gpucast/gl/primitives/bezierobject.hpp"

#include <fstream>
#include <regex>

#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/util/resource_factory.hpp>
#include <gpucast/gl/primitives/bezierobject_renderer.hpp>

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

    const float bezierobject::default_render_configuration::raycasting_error_tolerance = 0.001f;
    const float bezierobject::default_render_configuration::trimming_error_tolerance = 0.001f;
    const float bezierobject::default_render_configuration::tesselation_max_pixel_error = 1.5f;
    const float bezierobject::default_render_configuration::tesselation_max_pretesselation = 64.0f;
    const float bezierobject::default_render_configuration::tesselation_max_geometric_error = 0.0001f;

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::bezierobject(gpucast::beziersurfaceobject const& b)
      : _object(b),
      _trimming(b.trim_approach()),
      _rendermode(tesselation)
    {
      material m;
      m.randomize();

      _uniform_data.gpucast_material_ambient = m.ambient;
      _uniform_data.gpucast_material_diffuse = m.diffuse;
      _uniform_data.gpucast_material_specular = m.specular;

      _uniform_data.gpucast_shininess = m.shininess;
      _uniform_data.gpucast_opacity = m.opacity;

      _upload();

      trimming(b.trim_approach());
    }

    /////////////////////////////////////////////////////////////////////////////
    beziersurfaceobject const& bezierobject::object() const
    {
      return _object;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::init(unsigned subdiv_u, unsigned subdiv_v, unsigned preclass_resolution)
    {
      _object.init(subdiv_u, subdiv_v, preclass_resolution);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::draw(bezierobject_renderer& renderer)
    {
      //gpucast::gl::timer cputimer;
      //cputimer.start();

      switch (_rendermode)
      {
      case raycasting:
        _draw_by_raycasting(renderer);
        break;
      case tesselation:
      case shadow:
      case shadow_lowres:
        _draw_by_tesselation(renderer);
        break;
      }

      glFinish();
      glFlush();

      //cputimer.stop();
      //auto k = cputimer.result();
      //std::cout << "Object draw time : " << k.as_seconds() * 1000.0 << std::endl;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::raycasting_max_iterations(unsigned n)
    {
      _raycasting_max_iterations = n;
    }

    /////////////////////////////////////////////////////////////////////////////
    unsigned bezierobject::raycasting_max_iterations() const
    {
      return _raycasting_max_iterations;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::trimming_max_bisections(unsigned n)
    {
      _trimming_max_bisections = n;
    }
    /////////////////////////////////////////////////////////////////////////////
    unsigned bezierobject::trimming_max_bisections() const
    {
      return _trimming_max_bisections;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::raycasting_error_tolerance(float epsilon)
    {
      _raycasting_error_tolerance = epsilon;
    }

    /////////////////////////////////////////////////////////////////////////////
    float bezierobject::raycasting_error_tolerance() const
    {
      return _raycasting_error_tolerance;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::trimming_error_tolerance(float epsilon)
    {
      _trimming_error_tolerance = epsilon;
    }

    /////////////////////////////////////////////////////////////////////////////
    float bezierobject::trimming_error_tolerance() const
    {
      return _trimming_error_tolerance;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::tesselation_max_pixel_error(float px)
    {
      _tesselation_max_pixel_error = px;
    }

    /////////////////////////////////////////////////////////////////////////////
    float bezierobject::tesselation_max_pixel_error() const
    {
      return _tesselation_max_pixel_error;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::tesselation_max_geometric_error(float e)
    {
      _tesselation_max_geometric_error = e;
    }

    /////////////////////////////////////////////////////////////////////////////
    float bezierobject::tesselation_max_geometric_error() const
    {
      return _tesselation_max_geometric_error;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::tesselation_max_pretesselation(float t)
    {
      _tesselation_max_pretesselation = t;
    }

    /////////////////////////////////////////////////////////////////////////////
    float bezierobject::tesselation_max_pretesselation() const
    {
      return _tesselation_max_pretesselation;
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
    void bezierobject::enable_raycasting(bool enable)
    {
      _raycasting = enable;
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject::enable_raycasting() const
    {
      return _raycasting;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::rendermode(render_mode mode)
    {
      _rendermode = mode;
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::render_mode bezierobject::rendermode() const
    {
      return _rendermode;
    }

    /////////////////////////////////////////////////////////////////////////////
    beziersurfaceobject::trim_approach_t bezierobject::trimming() const
    {
      return _trimming;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::trimming(beziersurfaceobject::trim_approach_t approach)
    {
      if (_trimming != approach) {

        _trimming = approach;

        // update CPU object
        _object.trim_approach(_trimming);

        // upload to GPU
        _upload();
      }
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::fillmode(bezierobject::fill_mode mode)
    {
      _fill_mode = mode;
    }

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::fill_mode bezierobject::fillmode() const
    {
      return _fill_mode;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::set_material(material const& m)
    {
      _uniform_data.gpucast_material_ambient = m.ambient;
      _uniform_data.gpucast_material_diffuse = m.diffuse;
      _uniform_data.gpucast_material_specular = m.specular;

      _uniform_data.gpucast_shininess = m.shininess;
      _uniform_data.gpucast_opacity = m.opacity;
    }

    /////////////////////////////////////////////////////////////////////////////
    material bezierobject::get_material() const
    {
      return material{ _uniform_data.gpucast_material_ambient ,
        _uniform_data.gpucast_material_diffuse,
        _uniform_data.gpucast_material_specular,
        _uniform_data.gpucast_shininess,
        _uniform_data.gpucast_opacity };
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_draw_by_raycasting(bezierobject_renderer& renderer)
    {
      auto const& raycasting_program = renderer.get_raycasting_program();

      if (_culling) {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
      }
      else {
        glDisable(GL_CULL_FACE);
      }

      // draw proxy geometry
      _chull_vao.bind();
      _chull_indexarray.bind();

      renderer.begin_program(raycasting_program);

      // apply renderer-dependent uniforms 
      renderer.apply_uniforms(raycasting_program);

      _apply_uniforms(renderer, *raycasting_program, raycasting);

      glDrawElementsBaseVertex(GL_TRIANGLES, GLsizei(_size), GL_UNSIGNED_INT, 0, 0);

      renderer.end_program(raycasting_program);

      // unbind buffer
      _chull_indexarray.unbind();
      _chull_vao.unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_draw_by_tesselation(bezierobject_renderer& renderer)
    {
      auto const& pretesselation_program = renderer.get_pretesselation_program();
      auto const& tesselation_program = renderer.get_tesselation_program();
      auto tfbuffer = singleton<transform_feedback_buffer>::instance();

      if (!renderer.inside_frustum(*this)) {
        return;
      }

#define QUERY_XFB_PRIMITIVE_COUNT 0
#if QUERY_XFB_PRIMITIVE_COUNT
      unsigned primitive_query;
      glGenQueries(1, &primitive_query);
      glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, primitive_query);
#endif

      switch (_fill_mode) {
      case points:
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        break;
      case solid:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        break;
      case wireframe:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        break;
      }

      if (renderer.enable_pretessellation())
      {
        // configure transform feedback pass
        glEnable(GL_RASTERIZER_DISCARD);
        glPatchParameteri(GL_PATCH_VERTICES, 4);

        renderer.begin_program(pretesselation_program);
        {
          // bind VAO and draw
          _tesselation_vertex_array.bind();
          _tesselation_index_buffer.bind();

          // apply uniforms 
          _apply_uniforms(renderer, *pretesselation_program, tesselation);
          renderer.apply_uniforms(pretesselation_program);

          if (_tesselation_vertex_count == 0) {
            return;
          }

          tfbuffer->feedback->bind();
          tfbuffer->feedback->begin(GL_POINTS);
          {
            glDrawElements(GL_PATCHES, _tesselation_vertex_count, GL_UNSIGNED_INT, 0);
          }
          tfbuffer->feedback->end();
          tfbuffer->feedback->unbind();

          _tesselation_index_buffer.unbind();
          _tesselation_vertex_array.unbind();
        }
        renderer.end_program(pretesselation_program);
        glDisable(GL_RASTERIZER_DISCARD);
      }

#if QUERY_XFB_PRIMITIVE_COUNT
      glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
      int ready = GL_FALSE;
      while (ready != GL_TRUE) {
        glGetQueryObjectiv(primitive_query, GL_QUERY_RESULT_AVAILABLE, &ready);
      }
      std::cout << "Query ready: " << std::endl;
      int nprimitives = 0;
      glGetQueryObjectiv(primitive_query, GL_QUERY_RESULT, &nprimitives);

      BOOST_LOG_TRIVIAL(info) << "Primitives written : " << nprimitives << std::endl;
      glDeleteQueries(1, &primitive_query);
#endif

      renderer.begin_program(tesselation_program);
      {
        // apply uniforms to final tesselation program
        _apply_uniforms(renderer, *tesselation_program, tesselation);
        renderer.apply_uniforms(tesselation_program);

        // bind transform feedback buffer and draw
        if (renderer.enable_pretessellation()) {
          tfbuffer->vertex_array_object->bind();
          if (renderer.enable_triangular_tesselation()) {
            glPatchParameteri(GL_PATCH_VERTICES, 3);
          }
          glDrawTransformFeedback(GL_PATCHES, tfbuffer->feedback->id());
        }
        else {
          // bind VAO and draw
          glPatchParameteri(GL_PATCH_VERTICES, 4);
          _tesselation_vertex_array.bind();
          _tesselation_index_buffer.bind();
          glDrawElements(GL_PATCHES, _tesselation_vertex_count, GL_UNSIGNED_INT, 0);
          _tesselation_index_buffer.unbind();
          _tesselation_vertex_array.unbind();
        }
      }
      renderer.end_program(tesselation_program);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_apply_uniforms(bezierobject_renderer& renderer, program const& p, render_mode mode)
    {
#if 1
      _uniform_data.gpucast_enable_newton_iteration = _raycasting;
      _uniform_data.gpucast_raycasting_iterations = _raycasting_max_iterations;
      _uniform_data.gpucast_raycasting_error_tolerance = _raycasting_error_tolerance;
      _uniform_data.gpucast_tesselation_max_pixel_error = _tesselation_max_pixel_error;

      _uniform_data.gpucast_max_pre_tesselation = _tesselation_max_pretesselation;
      _uniform_data.gpucast_max_geometric_error = _tesselation_max_geometric_error;
      _uniform_data.gpucast_shadow_mode = _rendermode;
      _uniform_data.gpucast_trimming_max_bisections = _trimming_max_bisections;

      _uniform_data.gpucast_trimming_error_tolerance = _trimming_error_tolerance;
      _uniform_data.gpucast_trimming_method = _trimming;

      auto memptr = _uniform_block.map(GL_WRITE_ONLY);
      memcpy(memptr, (void*)&_uniform_data, sizeof(bezierobject_uniformbuffer_layout));
      _uniform_block.unmap();

      p.set_uniformbuffer("gpucast_object_uniforms", _uniform_block, gpucast::gl::bezierobject_renderer::GPUCAST_OBJECT_UBO_BINDINGPOINT);
#else

      // render parameters
      p.set_uniform1i("gpucast_enable_newton_iteration", _raycasting);
      p.set_uniform1i("gpucast_raycasting_iterations", _raycasting_max_iterations);
      p.set_uniform1f("gpucast_raycasting_error_tolerance", _raycasting_error_tolerance);
      p.set_uniform1f("gpucast_tesselation_max_pixel_error", _tesselation_max_pixel_error);
      p.set_uniform1f("gpucast_max_pre_tesselation", _tesselation_max_pretesselation);
      p.set_uniform1f("gpucast_max_geometric_error", _tesselation_max_geometric_error);
      p.set_uniform1i("gpucast_shadow_mode", _rendermode);
      p.set_uniform1i("gpucast_trimming_max_bisections", _trimming_max_bisections);
      p.set_uniform1f("gpucast_trimming_error_tolerance", _trimming_error_tolerance);
      p.set_uniform1i("gpucast_trimming_method", int(_trimming));

      // material properties
      p.set_uniform4f("gpucast_material_ambient", _material.ambient[0], _material.ambient[1], _material.ambient[2], 1.0);
      p.set_uniform4f("gpucast_material_diffuse", _material.diffuse[0], _material.diffuse[1], _material.diffuse[2], 1.0);
      p.set_uniform4f("gpucast_material_specular", _material.specular[0], _material.specular[1], _material.specular[2], 1.0);
      p.set_uniform1f("gpucast_shininess", _material.shininess);
      p.set_uniform1f("gpucast_opacity", _material.opacity);
#endif
      // data uniforms

      switch (mode)
      {
      case raycasting:
        p.set_texturebuffer("gpucast_control_point_buffer", _controlpoints, renderer.next_texunit());
        p.set_texturebuffer("gpucast_obb_buffer", _obbs, renderer.next_texunit());
        break;
      case tesselation:
        p.set_texturebuffer("gpucast_control_point_buffer", _controlpoints, renderer.next_texunit());
        p.set_texturebuffer("gpucast_obb_buffer", _obbs, renderer.next_texunit());
        p.set_shaderstoragebuffer("gpucast_attribute_ssbo", _tesselation_attribute_buffer, bezierobject_renderer::GPUCAST_ATTRIBUTE_SSBO_BINDING);
        break;
      }

      switch (_trimming)
      {
      case beziersurfaceobject::curve_binary_partition:
        p.set_texturebuffer("gpucast_bp_trimdata", _db_partition, renderer.next_texunit());
        p.set_texturebuffer("gpucast_bp_celldata", _db_celldata, renderer.next_texunit());
        p.set_texturebuffer("gpucast_bp_curvelist", _db_curvelist, renderer.next_texunit());
        p.set_texturebuffer("gpucast_bp_curvedata", _db_curvedata, renderer.next_texunit());
        p.set_texturebuffer("gpucast_preclassification", _db_preclassification, renderer.next_texunit());
        break;
      case beziersurfaceobject::contour_binary_partition:
        p.set_texturebuffer("gpucast_cmb_partition", _cmb_partition, renderer.next_texunit());
        p.set_texturebuffer("gpucast_cmb_contourlist", _cmb_contourlist, renderer.next_texunit());
        p.set_texturebuffer("gpucast_cmb_curvelist", _cmb_curvelist, renderer.next_texunit());
        p.set_texturebuffer("gpucast_cmb_curvedata", _cmb_curvedata, renderer.next_texunit());
        p.set_texturebuffer("gpucast_cmb_pointdata", _cmb_pointdata, renderer.next_texunit());
        p.set_texturebuffer("gpucast_preclassification", _cmb_preclassification, renderer.next_texunit());
        break;
      case beziersurfaceobject::contour_kd_partition:
        p.set_texturebuffer("gpucast_kd_partition", _kd_partition, renderer.next_texunit());
        p.set_texturebuffer("gpucast_kd_contourlist", _kd_contourlist, renderer.next_texunit());
        p.set_texturebuffer("gpucast_kd_curvelist", _kd_curvelist, renderer.next_texunit());
        p.set_texturebuffer("gpucast_kd_curvedata", _kd_curvedata, renderer.next_texunit());
        p.set_texturebuffer("gpucast_kd_pointdata", _kd_pointdata, renderer.next_texunit());
        p.set_texturebuffer("gpucast_preclassification", _kd_preclassification, renderer.next_texunit());
        break;
      case beziersurfaceobject::contour_list:

        p.set_shaderstoragebuffer("gpucast_loop_buffer", _loop_list_loops, 3);
        p.set_shaderstoragebuffer("gpucast_contour_buffer", _loop_list_contours, 4);
        p.set_shaderstoragebuffer("gpucast_curve_buffer", _loop_list_curves, 5);
        p.set_shaderstoragebuffer("gpucast_point_buffer", _loop_list_points, 6);

        p.set_texturebuffer("gpucast_preclassification", _loop_list_preclassification, renderer.next_texunit());
        break;
      case beziersurfaceobject::no_trimming:
        p.set_texturebuffer("gpucast_kd_partition", _kd_partition, renderer.next_texunit());
        p.set_texturebuffer("gpucast_preclassification", _kd_preclassification, renderer.next_texunit());
        break;
      };
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload()
    {
      _uniform_block.bufferdata(sizeof(bezierobject_uniformbuffer_layout), (void*)&_uniform_data, GL_DYNAMIC_DRAW);

      _upload_trimming_buffers();
      _upload_controlpoint_buffer();
      _upload_raycasting_buffers();
      _upload_tesselation_buffers();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload_trimming_buffers()
    {
      // contour partition with double binary partition
      auto cmb_serial = _object.serialized_trimdata_as_contour_binary();
      _cmb_partition.update(cmb_serial->partition.begin(), cmb_serial->partition.end());
      _cmb_contourlist.update(cmb_serial->contourlist.begin(), cmb_serial->contourlist.end());
      _cmb_curvelist.update(cmb_serial->curvelist.begin(), cmb_serial->curvelist.end());
      _cmb_curvedata.update(cmb_serial->curvedata.begin(), cmb_serial->curvedata.end());
      _cmb_pointdata.update(cmb_serial->pointdata.begin(), cmb_serial->pointdata.end());
      _cmb_preclassification.update(cmb_serial->preclassification.begin(), cmb_serial->preclassification.end());

      _cmb_partition.format(GL_RGBA32F);
      _cmb_contourlist.format(GL_RGBA32F);
      _cmb_curvelist.format(GL_RGBA32F);
      _cmb_curvedata.format(GL_R32F);
      _cmb_pointdata.format(GL_RGB32F);
      _cmb_preclassification.format(GL_R8UI);

      // contour map with kd tree partition
      auto kd_serial = _object.serialized_trimdata_as_contour_kd();
      _kd_partition.update(kd_serial->partition.begin(), kd_serial->partition.end());
      _kd_contourlist.update(kd_serial->contourlist.begin(), kd_serial->contourlist.end());
      _kd_curvelist.update(kd_serial->curvelist.begin(), kd_serial->curvelist.end());
      _kd_curvedata.update(kd_serial->curvedata.begin(), kd_serial->curvedata.end());
      _kd_pointdata.update(kd_serial->pointdata.begin(), kd_serial->pointdata.end());
      _kd_preclassification.update(kd_serial->preclassification.begin(), kd_serial->preclassification.end());

      _kd_partition.format(GL_RGBA32F);
      _kd_contourlist.format(GL_RGBA32F);
      _kd_curvelist.format(GL_RGBA32F);
      _kd_curvedata.format(GL_R32F);
      _kd_pointdata.format(GL_RGB32F);
      _kd_preclassification.format(GL_R8UI);

      // classic double binary partition
      auto db_serial = _object.serialized_trimdata_as_double_binary();
      _db_partition.update(db_serial->partition.begin(), db_serial->partition.end());
      _db_celldata.update(db_serial->celldata.begin(), db_serial->celldata.end());
      _db_curvelist.update(db_serial->curvelist.begin(), db_serial->curvelist.end());
      _db_curvedata.update(db_serial->curvedata.begin(), db_serial->curvedata.end());
      _db_preclassification.update(db_serial->preclassification.begin(), db_serial->preclassification.end());

      _db_partition.format(GL_RGBA32F);
      _db_celldata.format(GL_RGBA32F);
      _db_curvelist.format(GL_RGBA32F);
      _db_curvedata.format(GL_RGB32F);
      _db_preclassification.format(GL_R8UI);

      // contours as loop lists
      auto ll_serial = _object.serialized_trimdata_as_contour_loop_list();
      _loop_list_loops.update(ll_serial->loops.begin(), ll_serial->loops.end());
      _loop_list_contours.update(ll_serial->contours.begin(), ll_serial->contours.end());
      _loop_list_curves.update(ll_serial->curves.begin(), ll_serial->curves.end());
      _loop_list_points.update(ll_serial->points.begin(), ll_serial->points.end());
      _loop_list_preclassification.update(ll_serial->preclassification.begin(), ll_serial->preclassification.end());

      _loop_list_preclassification.format(GL_R8UI);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload_controlpoint_buffer()
    {
      // update texturebuffers
      _controlpoints.update(_object.serialized_controlpoints().begin(), _object.serialized_controlpoints().end());
      _controlpoints.format(GL_RGBA32F);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload_raycasting_buffers()
    {
      // update attribute buffers
      _chull_attribarray0.update(_object.serialized_raycasting_data_attrib0().begin(), _object.serialized_raycasting_data_attrib0().end());
      _chull_attribarray1.update(_object.serialized_raycasting_data_attrib1().begin(), _object.serialized_raycasting_data_attrib1().end());
      _chull_attribarray2.update(_object.serialized_raycasting_data_attrib2().begin(), _object.serialized_raycasting_data_attrib2().end());
      _chull_attribarray3.update(_object.serialized_raycasting_data_attrib3().begin(), _object.serialized_raycasting_data_attrib3().end());

      // update index array buffer
      _chull_indexarray.update(_object.serialized_raycasting_data_indices().begin(), _object.serialized_raycasting_data_indices().end());
      _size = _object.serialized_raycasting_data_indices().size();

      _obbs.update(_object.serialized_tesselation_obbs().begin(), _object.serialized_tesselation_obbs().end());
      _obbs.format(GL_RGBA32F);

      // bind vertex array object and reset vertexarrayoffsets
      _chull_vao.bind();
      {
        _chull_vao.attrib_array(_chull_attribarray0, 0, 3, GL_FLOAT, false, 0, 0);
        _chull_vao.enable_attrib(0);

        _chull_vao.attrib_array(_chull_attribarray1, 1, 4, GL_FLOAT, false, 0, 0);
        _chull_vao.enable_attrib(1);

        _chull_vao.attrib_array(_chull_attribarray2, 2, 4, GL_FLOAT, false, 0, 0);
        _chull_vao.enable_attrib(2);

        _chull_vao.attrib_array(_chull_attribarray3, 3, 4, GL_FLOAT, false, 0, 0);
        _chull_vao.enable_attrib(3);
      }
      // finally unbind vertex array object 
      _chull_vao.unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload_tesselation_buffers()
    {
      // tesselation vertex setup
      _tesselation_vertex_buffer.update(_object.serialized_tesselation_domain_buffer().begin(), _object.serialized_tesselation_domain_buffer().end());
      _tesselation_index_buffer.update(_object.serialized_tesselation_index_buffer().begin(), _object.serialized_tesselation_index_buffer().end());
      _tesselation_vertex_count = _object.serialized_tesselation_index_buffer().size();

      // create vertex array object bindings
      int stride = sizeof(math::vec3f) + sizeof(unsigned) + sizeof(math::vec4f);

      _tesselation_vertex_array.bind();
      {
        _tesselation_vertex_array.attrib_array(_tesselation_vertex_buffer, 0, 3, GL_FLOAT, false, stride, 0);
        _tesselation_vertex_array.attrib_array(_tesselation_vertex_buffer, 1, 1, GL_UNSIGNED_INT, false, stride, sizeof(gpucast::math::vec3f));
        _tesselation_vertex_array.attrib_array(_tesselation_vertex_buffer, 2, 4, GL_FLOAT, false, stride, sizeof(gpucast::math::vec3f) + sizeof(unsigned));

        _tesselation_vertex_array.enable_attrib(0);
        _tesselation_vertex_array.enable_attrib(1);
        _tesselation_vertex_array.enable_attrib(2);
      }
      _tesselation_vertex_array.unbind();

      gpucast::gl::hullvertexmap hvm;
      _tesselation_hullvertexmap.update(hvm.data.begin(), hvm.data.end());
      _tesselation_attribute_buffer.update(_object.serialized_tesselation_attribute_data().begin(), _object.serialized_tesselation_attribute_data().end());  
    }

  } // namespace gl
} // namespace gpucast 

