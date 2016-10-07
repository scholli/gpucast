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

    const float bezierobject::default_render_configuration::raycasting_error_tolerance     = 0.001f;
    const float bezierobject::default_render_configuration::trimming_error_tolerance       = 0.001f;
    const float bezierobject::default_render_configuration::tesselation_max_pixel_error    = 4.0f;
    const float bezierobject::default_render_configuration::tesselation_max_pretesselation = 64.0f;

    /////////////////////////////////////////////////////////////////////////////
    bezierobject::bezierobject(gpucast::beziersurfaceobject const& b)
      : _object(b),
        _trimming(b.trim_approach()),
        _rendermode(tesselation)
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
    void bezierobject::draw()
    {
      switch (_rendermode) 
      {
        case raycasting:
          _draw_by_raycasting();
          break;
        case tesselation:
        case shadow:
        case shadow_lowres:
          _draw_by_tesselation();
          break;
      }
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

#define QUERY_XFB_PRIMITIVE_COUNT 0
#if QUERY_XFB_PRIMITIVE_COUNT
      unsigned primitive_query;
      glGenQueries(1, &primitive_query);
      glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, primitive_query);
#endif

      // configure transform feedback pass
      glEnable(GL_RASTERIZER_DISCARD);
      glPatchParameteri(GL_PATCH_VERTICES, 4);

      switch (_fill_mode) {
      case points : 
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        break;
      case solid :
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        break;
      case wireframe :
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        break;
      }

      renderer->begin_program(pretesselation_program);
      {
        // bind VAO and draw
        _tesselation_vertex_array.bind();
        _tesselation_index_buffer.bind();   

        // apply uniforms 
        _apply_uniforms(*pretesselation_program, tesselation);
        renderer->apply_uniforms(pretesselation_program);

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
      renderer->end_program(pretesselation_program);

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

      glDisable(GL_RASTERIZER_DISCARD);
      renderer->begin_program(tesselation_program);
      {
        // apply uniforms to final tesselation program
        _apply_uniforms(*tesselation_program, tesselation);
        renderer->apply_uniforms(tesselation_program);

        // bind transform feedback buffer and draw
        tfbuffer->vertex_array_object->bind();
        glDrawTransformFeedback(GL_PATCHES, tfbuffer->feedback->id());
      }
      renderer->end_program(tesselation_program);
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject::_apply_uniforms(program const& p, render_mode mode)
    {
      // render parameters
      p.set_uniform1i("gpucast_enable_newton_iteration", _raycasting);
      p.set_uniform1i("gpucast_raycasting_iterations", _raycasting_max_iterations);
      p.set_uniform1f("gpucast_raycasting_error_tolerance", _raycasting_error_tolerance);
      p.set_uniform1f("gpucast_tesselation_max_error", _tesselation_max_pixel_error);
      p.set_uniform1f("gpucast_max_pre_tesselation", _tesselation_max_pretesselation);
      p.set_uniform1i("gpucast_shadow_mode", _rendermode);

      p.set_uniform1i("gpucast_trimming_max_bisections", _trimming_max_bisections);
      p.set_uniform1f("gpucast_trimmig_error_tolerance", _trimming_error_tolerance);
      p.set_uniform1i("gpucast_trimming_method", int(_trimming) );

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
        break;
      case tesselation : 
        
        p.set_texturebuffer("gpucast_parametric_buffer", _tesselation_parametric_texture_buffer, renderer->next_texunit());
        p.set_texturebuffer("gpucast_obb_buffer", _obbs, renderer->next_texunit());
        p.set_shaderstoragebuffer("gpucast_attribute_ssbo", _tesselation_attribute_buffer, bezierobject_renderer::GPUCAST_ATTRIBUTE_SSBO_BINDING);
        break;
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

        p.set_shaderstoragebuffer("gpucast_loop_buffer", _loop_list_loops, 3);
        p.set_shaderstoragebuffer("gpucast_contour_buffer", _loop_list_contours, 4);
        p.set_shaderstoragebuffer("gpucast_curve_buffer", _loop_list_curves, 5);
        p.set_shaderstoragebuffer("gpucast_point_buffer", _loop_list_points, 6);

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
      _tesselation_parametric_texture_buffer.update(_object.serialized_tesselation_control_point_buffer().begin(), _object.serialized_tesselation_control_point_buffer().end());
      _tesselation_parametric_texture_buffer.format(GL_RGBA32F);

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

