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
    bezierobject::bezierobject(gpucast::beziersurfaceobject const& b)
      : _object(b),
        _trimming(b.trim_approach())
    {
      _material.randomize();

      _upload();

      trimming(b.trim_approach());
    }

    /////////////////////////////////////////////////////////////////////////////
    beziersurfaceobject const& bezierobject::object() const
    {
      return _object;
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::draw(bezierobject::render_mode mode)
    {
      switch (mode) {
      case raycasting:
        _draw_by_raycasting();
        break;
      case tesselation:
        _draw_by_tesselation();
        break;
      }
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
    void bezierobject::max_trimming_bisections(unsigned n)
    {
      _max_trimming_bisections = n;
    }
    /////////////////////////////////////////////////////////////////////////////
    unsigned bezierobject::max_trimming_bisections() const
    {
      return _max_trimming_bisections;
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
    void bezierobject::_draw_by_raycasting()
    {
      auto renderer = bezierobject_renderer::instance();
      auto const& raycasting_program = renderer->get_raycasting_program();

      // draw proxy geometry
      _chull_vao.bind();
      _chull_indexarray.bind();

      renderer->begin_program(raycasting_program);

      // apply renderer-dependent uniforms 
      renderer->apply_uniforms(raycasting_program);

      _apply_uniforms(*raycasting_program, raycasting);

      glDrawElementsBaseVertex(GL_TRIANGLES, GLsizei(_size), GL_UNSIGNED_INT, 0, 0);

      renderer->end_program(raycasting_program);

      // unbind buffer
      _chull_indexarray.unbind();
      _chull_vao.unbind();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_draw_by_tesselation()
    {     
      auto renderer = bezierobject_renderer::instance();
      auto const& pretesselation_program = renderer->get_pretesselation_program();
      auto const& tesselation_program = renderer->get_tesselation_program();
      auto tfbuffer = singleton<transform_feedback_buffer>::instance();
   
      gl::error("begin draw xfb: ");

#define QUERY_XFB_PRIMITIVE_COUNT 1
#if QUERY_XFB_PRIMITIVE_COUNT
      unsigned primitive_query;
      glGenQueries(1, &primitive_query);
      glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, primitive_query);
#endif

      // configure transform feedback pass
      glEnable(GL_RASTERIZER_DISCARD);
      glPatchParameteri(GL_PATCH_VERTICES, 4);

      renderer->begin_program(pretesselation_program);
      {
        // bind VAO and draw
        _tesselation_vertex_array.bind();
        _tesselation_index_buffer.bind();   

        // apply uniforms 
        _apply_uniforms(*pretesselation_program, tesselation);
        renderer->apply_uniforms(pretesselation_program);

        BOOST_LOG_TRIVIAL(info) << "draw patches : " << _tesselation_vertex_count << std::endl;
        tfbuffer->feedback->bind();
        tfbuffer->feedback->begin(GL_POINTS);
        {
          gl::error("before draw elements: ");
          glDrawElements(GL_PATCHES, _tesselation_vertex_count, GL_UNSIGNED_INT, 0);
        }
        tfbuffer->feedback->end();
        tfbuffer->feedback->unbind();

        _tesselation_index_buffer.unbind();
        _tesselation_vertex_array.unbind();
      }
      renderer->end_program(pretesselation_program);

      gl::error("end xfb: ");
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

      gl::error("begin draw: ");
      glDisable(GL_RASTERIZER_DISCARD);
      gl::error("begin A: ");
      renderer->begin_program(tesselation_program);
      {
        gl::error("begin B: ");
        // apply uniforms to final tesselation program
        _apply_uniforms(*tesselation_program, tesselation);
        gl::error("begin C2: ");
        renderer->apply_uniforms(tesselation_program);

        gl::error("begin C: ");
        // bind transform feedback buffer and draw
        tfbuffer->vertex_array_object->bind();
        glDrawTransformFeedback(GL_PATCHES, tfbuffer->feedback->id());
        gl::error("begin D: ");
      }
      renderer->end_program(tesselation_program);
      gl::error("end draw: ");
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_apply_uniforms(program const& p, render_mode mode)
    {
      BOOST_LOG_TRIVIAL(info) << "bezierobject::_apply_uniforms()" << std::endl;

      // render parameters
      p.set_uniform1i("gpucast_raycasting_iterations", _iterations);
      p.set_uniform1i("gpucast_trimming_max_bisections", _max_trimming_bisections);
      p.set_uniform1i("gpucast_trimming_method", int(_trimming) );
      p.set_uniform1i("gpucast_enable_newton_iteration", _raycasting);
      p.set_uniform1f("gpucast_raycasting_error_tolerance", _epsilon);

      // material properties
      p.set_uniform3f("mat_ambient", _material.ambient[0], _material.ambient[1], _material.ambient[2]);
      p.set_uniform3f("mat_diffuse", _material.diffuse[0], _material.diffuse[1], _material.diffuse[2]);
      p.set_uniform3f("mat_specular", _material.specular[0], _material.specular[1], _material.specular[2]);
      p.set_uniform1f("shininess", _material.shininess);
      p.set_uniform1f("opacity", _material.opacity);

      // data uniforms
      auto renderer = bezierobject_renderer::instance();
      switch (mode)
      {
      case raycasting :
        p.set_texturebuffer("gpucast_control_point_buffer", _controlpoints, renderer->next_texunit());
        p.set_texturebuffer("gpucast_obb_buffer", _obbs, renderer->next_texunit());
      case tesselation : 
        p.set_texturebuffer("gpucast_parametric_buffer", _tesselation_parametric_texture_buffer, renderer->next_texunit());
        p.set_texturebuffer("gpucast_obb_buffer", _obbs, renderer->next_texunit());
        p.set_shaderstoragebuffer("gpucast_attribute_ssbo", _tesselation_attribute_buffer, bezierobject_renderer::GPUCAST_ATTRIBUTE_SSBO_BINDING);
      }

      switch (_trimming) 
      {
      case beziersurfaceobject::curve_binary_partition:
        p.set_texturebuffer("gpucast_bp_trimdata", _db_partition, renderer->next_texunit());
        p.set_texturebuffer("gpucast_bp_celldata", _db_celldata, renderer->next_texunit());
        p.set_texturebuffer("gpucast_bp_curvelist", _db_curvelist, renderer->next_texunit());
        p.set_texturebuffer("gpucast_bp_curvedata", _db_curvedata, renderer->next_texunit());
        p.set_texturebuffer("gpucast_preclassification", _db_preclassification, renderer->next_texunit());
        break;
      case beziersurfaceobject::contour_binary_partition:
        p.set_texturebuffer("gpucast_cmb_partition", _cmb_partition, renderer->next_texunit());
        p.set_texturebuffer("gpucast_cmb_contourlist", _cmb_contourlist, renderer->next_texunit());
        p.set_texturebuffer("gpucast_cmb_curvelist", _cmb_curvelist, renderer->next_texunit());
        p.set_texturebuffer("gpucast_cmb_curvedata", _cmb_curvedata, renderer->next_texunit());
        p.set_texturebuffer("gpucast_cmb_pointdata", _cmb_pointdata, renderer->next_texunit());
        p.set_texturebuffer("gpucast_preclassification", _cmb_preclassification, renderer->next_texunit());
        break;
      case beziersurfaceobject::contour_kd_partition:
        p.set_texturebuffer("gpucast_kd_partition", _kd_partition, renderer->next_texunit());
        p.set_texturebuffer("gpucast_kd_contourlist", _kd_contourlist, renderer->next_texunit());
        p.set_texturebuffer("gpucast_kd_curvelist", _kd_curvelist, renderer->next_texunit());
        p.set_texturebuffer("gpucast_kd_curvedata", _kd_curvedata, renderer->next_texunit());
        p.set_texturebuffer("gpucast_kd_pointdata", _kd_pointdata, renderer->next_texunit());
        p.set_texturebuffer("gpucast_preclassification", _kd_preclassification, renderer->next_texunit());
        break;
      case beziersurfaceobject::contour_list:
        _loop_list_loops.bind_buffer_base(1);
        p.set_shaderstoragebuffer("gpucast_loop_buffer", _loop_list_loops, 1);

        _loop_list_contours.bind_buffer_base(2);
        p.set_shaderstoragebuffer("gpucast_contour_buffer", _loop_list_contours, 2);

        _loop_list_curves.bind_buffer_base(3);
        p.set_shaderstoragebuffer("gpucast_curve_buffer", _loop_list_curves, 3);

        _loop_list_points.bind_buffer_base(4);
        p.set_shaderstoragebuffer("gpucast_point_buffer", _loop_list_points, 4);

        p.set_texturebuffer("gpucast_preclassification", _loop_list_preclassification, renderer->next_texunit());
        break;
      };
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload()
    {
      _upload_trimming_buffers();
      _upload_raycasting_buffers();
      _upload_tesselation_buffers();
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_upload_trimming_buffers()
    {
      BOOST_LOG_TRIVIAL(info) << "bezierobject::_upload_trimming_buffers()" << std::endl;

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
    void bezierobject::_upload_raycasting_buffers()
    {
      BOOST_LOG_TRIVIAL(info) << "bezierobject::_upload_raycasting_buffers()" << std::endl;

      // update attribute buffers
      _chull_attribarray0.update(_object.serialized_raycasting_data_attrib0().begin(), _object.serialized_raycasting_data_attrib0().end());
      _chull_attribarray1.update(_object.serialized_raycasting_data_attrib1().begin(), _object.serialized_raycasting_data_attrib1().end());
      _chull_attribarray2.update(_object.serialized_raycasting_data_attrib2().begin(), _object.serialized_raycasting_data_attrib2().end());
      _chull_attribarray3.update(_object.serialized_raycasting_data_attrib3().begin(), _object.serialized_raycasting_data_attrib3().end());

      // update index array buffer
      _chull_indexarray.update(_object.serialized_raycasting_data_indices().begin(), _object.serialized_raycasting_data_indices().end());
      _size = _object.serialized_raycasting_data_indices().size();

      // update texturebuffers
      _controlpoints.update(_object.serialized_raycasting_data_controlpoints().begin(), _object.serialized_raycasting_data_controlpoints().end());
      _controlpoints.format(GL_RGBA32F);

      _obbs.update(_object.serialized_raycasting_data_obbs().begin(), _object.serialized_raycasting_data_obbs().end());
      _obbs.format(GL_RGBA32F);

      int p = 0;
      BOOST_LOG_TRIVIAL(info) << "obb_buffer: " << std::endl;
      for (auto i : _object.serialized_raycasting_data_obbs()) {
        BOOST_LOG_TRIVIAL(info) << p++ << " : " << i << std::endl;
      }

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
      BOOST_LOG_TRIVIAL(info) << "bezierobject::_upload_tesselation_buffers()" << std::endl;

      BOOST_LOG_TRIVIAL(info) << "control points size : " << _object.serialized_tesselation_control_point_buffer().size() << std::endl;
      BOOST_LOG_TRIVIAL(info) << "tesselation data size : " << _object.serialized_tesselation_attribute_data().size() << std::endl;
      BOOST_LOG_TRIVIAL(info) << "domain buffer size : " << _object.serialized_tesselation_domain_buffer().size() << std::endl;
      BOOST_LOG_TRIVIAL(info) << "vertex buffer size : " << _object.serialized_tesselation_domain_buffer().size() << std::endl;
      BOOST_LOG_TRIVIAL(info) << "index buffer size : " << _object.serialized_tesselation_index_buffer().size() << std::endl;

      // upload tesselation texture buffers
      std::cout << "parameter data : " << std::endl;
      for (auto i : _object.serialized_tesselation_control_point_buffer()) {
        std::cout << i << std::endl;
      }

      std::cout << "attribute data : " << std::endl;
      for (auto i : _object.serialized_tesselation_attribute_data()) {
        std::cout << "surface_offset = " << i.surface_offset << std::endl;
        std::cout << "order_u = " << i.order_u << std::endl;
        std::cout << "order_v = " << i.order_v << std::endl;
        std::cout << "trim_id = " << i.trim_id << std::endl;
        std::cout << "obb_id = " << i.obb_id << std::endl;
        std::cout << "nurbs_domain = " << i.nurbs_domain << std::endl;
        std::cout << "bbox_min = " << i.bbox_min << std::endl;
        std::cout << "bbox_max = " << i.bbox_max << std::endl;
      }

      std::cout << "domain buffer (vertex buffer) data : " << std::endl;
      for (auto i : _object.serialized_tesselation_domain_buffer()) {
        std::cout << i << std::endl;
      }

      std::cout << "index buffer data : " << std::endl;
      for (auto i : _object.serialized_tesselation_index_buffer()) {
        std::cout << i << std::endl;
      }

      _tesselation_parametric_texture_buffer.update(_object.serialized_tesselation_control_point_buffer().begin(), _object.serialized_tesselation_control_point_buffer().end());
      _tesselation_parametric_texture_buffer.format(GL_RGBA32F);

      //////////////
      ////////////// TMP
      auto float_to_uint = [](float const & i) { return *((unsigned*)(&i)); };

      std::cout << "VBO : " << std::endl;
      for (auto v : _object.serialized_tesselation_domain_buffer()) {
        if (v[3] != 0) {
          std::cout << v << std::endl;
          std::cout << "id = " << float_to_uint(v[3]) << std::endl;
        } else {
          std::cout << v << std::endl;
        }
      }
      ////////////// TMP
      //////////////

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



    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////

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
      BOOST_LOG_TRIVIAL(info) << "bezierobject_renderer::apply_uniforms()" << std::endl;
      // view parameters
      p->set_uniform1f("gpucast_clip_near", _nearplane);
      p->set_uniform1f("gpucast_clip_far", _farplane);

      // TODO: parameterize
      p->set_uniform1f("gpucast_tesselation_max_error", 4.0f);
      p->set_uniform1f("gpucast_max_pre_tesselation", 16.0f);
      p->set_uniform1i("gpucast_shadow_mode", 0);
      p->set_uniform2i("gpucast_resolution", _resolution[0], _resolution[1]);

      if (p == _pretesselation_program || p == _tesselation_program) {
        p->set_shaderstoragebuffer("gpucast_hullvertexmap_ssbo", *_hullvertexmap, GPUCAST_HULLVERTEXMAP_SSBO_BINDING);
      }

      // camera block
      p->set_uniform_matrix4fv("gpucast_projection_matrix", 1, false, &_projectionmatrix[0]);
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
      BOOST_LOG_TRIVIAL(info) << "_init_tesselation_program()" << std::endl;
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
      BOOST_LOG_TRIVIAL(info) << "_init_hullvertexmap()" << std::endl;

      _hullvertexmap = std::make_shared<shaderstoragebuffer>();

      hullvertexmap hvm;

      _hullvertexmap->update(hvm.data.begin(), hvm.data.end());
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_prefilter(unsigned prefilter_resolution) 
    {
      BOOST_LOG_TRIVIAL(info) << "_init_prefilter()" << std::endl;

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
      BOOST_LOG_TRIVIAL(info) << "_init_transform_feedback()" << std::endl;

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
        int stride = sizeof(math::vec3f) + sizeof(unsigned) + sizeof(math::vec2f) + sizeof(float);
        tfbuffer->vertex_array_object->bind();
        {
          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 0, 3, GL_FLOAT, false, stride, 0);
          tfbuffer->vertex_array_object->enable_attrib(0);

          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 1, 1, GL_UNSIGNED_INT, false, stride, sizeof(gpucast::math::vec3f));
          tfbuffer->vertex_array_object->enable_attrib(1);

          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 2, 2, GL_FLOAT, false, stride, sizeof(gpucast::math::vec3f) + sizeof(unsigned));
          tfbuffer->vertex_array_object->enable_attrib(2);

          tfbuffer->vertex_array_object->attrib_array(*tfbuffer->buffer, 3, 1, GL_FLOAT, false, stride, sizeof(gpucast::math::vec3f) + sizeof(unsigned) + sizeof(gpucast::math::vec2f));
          tfbuffer->vertex_array_object->enable_attrib(3);
        }
        tfbuffer->vertex_array_object->unbind();
      }
    }

  } // namespace gl
} // namespace gpucast 

