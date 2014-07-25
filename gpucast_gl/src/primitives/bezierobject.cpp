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
    void bezierobject::trimming(bool enable)
    {
      _trimming = enable;
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject::trimming() const
    {
      return _trimming;
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
    void bezierobject::trim_approach(beziersurfaceobject::trimapproach type)
    {
      _trim_approach = type;
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
      p.set_uniform1i("trimming_enabled", _trimming);
      p.set_uniform1i("raycasting_enabled", _raycasting);
      p.set_uniform1f("epsilon_object_space", _epsilon);
      p.set_uniform1i("trimapproach", int(_trim_approach));

      // material properties
      p.set_uniform3f("mat_ambient", _material.ambient[0], _material.ambient[1], _material.ambient[2]);
      p.set_uniform3f("mat_diffuse", _material.diffuse[0], _material.diffuse[1], _material.diffuse[2]);
      p.set_uniform3f("mat_specular", _material.specular[0], _material.specular[1], _material.specular[2]);
      p.set_uniform1f("shininess", _material.shininess);
      p.set_uniform1f("opacity", _material.opacity);

      // data uniforms
      auto& renderer = bezierobject_renderer::instance();
      p.set_texturebuffer("vertexdata", _controlpoints, renderer.next_texunit());

      p.set_texturebuffer("cmb_partition", _cmb_partition, renderer.next_texunit());
      p.set_texturebuffer("cmb_contourlist", _cmb_contourlist, renderer.next_texunit());
      p.set_texturebuffer("cmb_curvelist", _cmb_contourlist, renderer.next_texunit());
      p.set_texturebuffer("cmb_curvedata", _cmb_curvelist, renderer.next_texunit());
      p.set_texturebuffer("cmb_pointdata", _cmb_pointdata, renderer.next_texunit());

      p.set_texturebuffer("bp_trimdata", _db_partition, renderer.next_texunit());
      p.set_texturebuffer("bp_celldata", _db_celldata, renderer.next_texunit());
      p.set_texturebuffer("bp_curvelist", _db_curvelist, renderer.next_texunit());
      p.set_texturebuffer("bp_curvedata", _db_curvedata, renderer.next_texunit());

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
      _controlpoints.format(GL_RGBA32F_ARB);

      _cmb_partition.update(b._cmb_partition.begin(), b._cmb_partition.end());
      _cmb_contourlist.update(b._cmb_contourlist.begin(), b._cmb_contourlist.end());
      _cmb_curvelist.update(b._cmb_curvelist.begin(), b._cmb_curvelist.end());
      _cmb_curvedata.update(b._cmb_curvedata.begin(), b._cmb_curvedata.end());
      _cmb_pointdata.update(b._cmb_pointdata.begin(), b._cmb_pointdata.end());

      _cmb_partition.format(GL_RGBA32F);
      _cmb_contourlist.format(GL_RG32F);
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
      _init_program();
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
    void bezierobject_renderer::modelviewmatrix(gpucast::gl::matrix4f const& m)
    {
      _modelviewmatrix = m;

      _modelviewmatrixinverse = m;
      _modelviewmatrixinverse.invert();

      _normalmatrix = m.normalmatrix();
      _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::projectionmatrix(gpucast::gl::matrix4f const& m)
    {
      _projectionmatrix = m;

      _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
      _modelviewprojectionmatrixinverse = gpucast::gl::inverse(_modelviewprojectionmatrix);
    }


    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_nearfar(float near, float far)
    {
      _nearplane = near;
      _farplane = far;
      projectionmatrix(gpucast::gl::frustum(-1.0f, 1.0f, -1.0f, 1.0f, _nearplane, _farplane));
    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::set_background(gpucast::gl::vec3f const& color)
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

    }

    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::_init_program()
    {
      _program.reset(new program);

      init_program(_program,
        "./gpucast_core/glsl/trimmed_surface/raycast_surface.glsl.vert",
        "./gpucast_core/glsl/trimmed_surface/raycast_surface.glsl.frag");
    }

    /////////////////////////////////////////////////////////////////////////////
    bool bezierobject_renderer::_path_to_file(std::string const& filename, std::string& result) const
    {
      std::set<std::string>::const_iterator path_to_source = _pathlist.begin();
        
      while (path_to_source != _pathlist.end())
      {
        if (boost::filesystem::exists((*path_to_source) + filename)) {
          break;
        }
        ++path_to_source;
      }

      if (path_to_source != _pathlist.end())
      {
        std::fstream fstr(((*path_to_source) + filename).c_str(), std::ios::in);
        if (fstr)
        {
          result = std::string((std::istreambuf_iterator<char>(fstr)), std::istreambuf_iterator<char>());

          std::smatch include_line;
          std::regex include_line_exp("#include[^\n]+");

          // as long as there are include lines
          while (std::regex_search(result, include_line, include_line_exp))
          {
            std::string include_line_str = include_line.str();
            std::smatch include_filename;
            std::regex include_filename_exp("[<\"][^>\"]+[>\"]");

            // extract include filename
            if (std::regex_search(include_line_str, include_filename, include_filename_exp))
            {
              std::string filename = include_filename.str();
              auto is_delimiter = [](char a) { return (a == '"') || (a == '<') || (a == '>'); };
              filename.erase(std::remove_if(filename.begin(), filename.end(), is_delimiter), filename.end());

              // replace include with source
              std::string included_source;
              bool success = _path_to_file(filename, included_source);
                
              if (success) {
                result.replace(result.find(include_line.str()), include_line.str().size(), included_source);
              }
              else {
                result.replace(result.find(include_line.str()), include_line.str().size(), "");
                std::cerr << "renderer::_path_to_file(): Could not open " << filename << std::endl;
              }
            }
          }
        }
        fstr.close();

        return true;
      }
      else {
        return false;
      }
    }



    /////////////////////////////////////////////////////////////////////////////
    void bezierobject_renderer::init_program(std::shared_ptr<gpucast::gl::program>& p,
                                             std::string const&                     vertexshader_filename,
                                             std::string const&                     fragmentshader_filename)
    {
      try {
        vertexshader     vs;
        fragmentshader   fs;

        p.reset(new program);

        std::string vs_source; 
        std::string fs_source; 
        std::string gs_source;

        bool vs_available = _path_to_file(vertexshader_filename, vs_source);
        bool fs_available = _path_to_file(fragmentshader_filename, fs_source);

        if (vs_available)
        {
          vs.set_source(vs_source.c_str());
          vs.compile();
          if (!vs.log().empty()) {
            std::fstream ostr(boost::filesystem::basename(vertexshader_filename) + ".vert.log", std::ios::out);
            ostr << vs.log() << std::endl;
            ostr.close();
          }
          p->add(&vs);
        }
        else {
          throw std::runtime_error("bezierobject_renderer::init_program (): Couldn't open file " + vertexshader_filename);
        }

        if (fs_available)
        {
          fs.set_source(fs_source.c_str());
          fs.compile();

          if (!fs.log().empty()) {
            std::fstream ostr(boost::filesystem::basename(fragmentshader_filename) + ".frag.log", std::ios::out);
            ostr << vs.log() << std::endl;
            ostr.close();
          }
          p->add(&fs);
        }
        else {
          throw std::runtime_error("bezierobject_renderer::init_program (): Couldn't open file " + fragmentshader_filename);
        }

        // link all shaders
        p->link();

        if (!p->log().empty())
        {
          // stream log to std output
          std::cout << " program log : " << p->log() << std::endl;
        }
        else {
          std::cout << "Compiling " << vertexshader_filename << ", " << fragmentshader_filename << " succeeded\n";
        }
      }
      catch (std::exception& e) {
        std::cerr << "renderer::init_program(): failed to init program : " << vertexshader_filename << ", " << fragmentshader_filename << "( " << e.what() << ")\n";
      }
    }


  } // namespace gl
} // namespace gpucast 
