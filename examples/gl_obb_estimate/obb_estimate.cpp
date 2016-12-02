/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : simple.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>
#include <vector>
#include <algorithm>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/atomicbuffer.hpp>
#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/shaderstoragebuffer.hpp>

#include <gpucast/gl/primitives/obb.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/hullvertexmap.hpp>

#include <gpucast/math/vec3.hpp>
#include <gpucast/math/vec4.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <gpucast/math/oriented_boundingbox.hpp>

#define HULLVERTEXMAP_SSBO_BINDING 0
#define ATOMIC_COUNTER_BINDING 1
#define ESTIMATE_COUNTER_BINDING 2
#define OBB_TEXTURE_UNIT 0

static const int window_width = 1920;
static const int window_height = 1080;

const gpucast::math::vec4f translation{ 0.4f, 1.0f, -1.4f, 1.0f };
const gpucast::math::vec3f rotation{ 0.5f, 0.4f, 0.7f };
const gpucast::math::vec4f scale{ 1.0f, 2.0f, 0.5f, 0.0f };

template <typename T>
T clamp(T const& a, T const& lower, T const& upper) {
  return std::min(std::max(a, lower), upper);
}

class application
{
public :

  application()
    : _program  (),
      _cube     (0, -1, 2, 1),
      _trackball(new gpucast::gl::trackball)
  {
    init_shader(); 
    init_obb();

    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));

    glEnable(GL_DEPTH_TEST);
  }
 
  ~application()  
  {}   
    
  void init_obb()
  {
    // create unit cube
    std::vector<gpucast::math::vec4f> vertices = { gpucast::math::vec4f(-1.0f, -1.0f, -1.0f, 1.0f),
       gpucast::math::vec4f(1.0f, -1.0f, -1.0f, 1.0f),
       gpucast::math::vec4f(-1.0f, 1.0f, -1.0f, 1.0f),
       gpucast::math::vec4f(1.0f, 1.0f, -1.0f, 1.0f),
       gpucast::math::vec4f(-1.0f, -1.0f, 1.0f, 1.0f),
       gpucast::math::vec4f(1.0f, -1.0f, 1.0f, 1.0f),
       gpucast::math::vec4f(-1.0f, 1.0f, 1.0f, 1.0f),
       gpucast::math::vec4f(1.0f, 1.0f, 1.0f, 1.0f) };

    // create unit cube
    std::vector<gpucast::math::vec4f> colors = { gpucast::math::vec4f(0.0f, 0.0f, 0.0f, 1.0f),
                                                 gpucast::math::vec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                                 gpucast::math::vec4f(0.0f, 1.0f, 0.0f, 1.0f),
                                                 gpucast::math::vec4f(1.0f, 1.0f, 0.0f, 1.0f),
                                                 gpucast::math::vec4f(0.0f, 0.0f, 1.0f, 1.0f),
                                                 gpucast::math::vec4f(1.0f, 0.0f, 1.0f, 1.0f),
                                                 gpucast::math::vec4f(0.0f, 1.0f, 1.0f, 1.0f),
                                                 gpucast::math::vec4f(1.0f, 1.0f, 1.0f, 1.0f) };

    // create transformation
    auto rotation_matrix = gpucast::math::make_rotation_x(rotation[0]) *
      gpucast::math::make_rotation_y(rotation[1]) *
      gpucast::math::make_rotation_z(rotation[2]);

    auto rotation_inverse_matrix = gpucast::math::inverse(rotation_matrix);

    auto model_matrix = gpucast::math::make_translation(translation[0], translation[1], translation[2]) *
      rotation_matrix * 
      gpucast::math::make_scale(scale[0], scale[1], scale[2]);

    // transform OBB
    for (auto& v : vertices) {
      v = model_matrix * v;
    }
  
    _cube.set_vertices(vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7]);
    _cube.set_colors(colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], colors[6], colors[7]);

    std::vector<float> rot_mat_data = { rotation_matrix[0], rotation_matrix[4], rotation_matrix[8],
                                        rotation_matrix[1], rotation_matrix[5], rotation_matrix[9],
                                        rotation_matrix[2], rotation_matrix[6], rotation_matrix[10] };
    gpucast::math::matrix<float, 3, 3> rot_mat(rot_mat_data.begin(), rot_mat_data.end());

    // serialize OBB for estimate
    gpucast::math::oriented_boundingbox<gpucast::math::point3f> obb (rot_mat, translation, -1.0f * scale, scale);
    _obbdata = obb.serialize<float>();
    _obb.update(_obbdata.begin(), _obbdata.end());
    _obb.format(GL_RGBA32F);

    gpucast::gl::hullvertexmap hvm;
    _hvm.update(hvm.data.begin(), hvm.data.end());

    // allocate memory for GPU feedback
    _estimate.bufferdata(100 * sizeof(unsigned), 0, GL_DYNAMIC_COPY);
  }

  void init_shader()  
  {
    std::string vertexshader_code = R"(
     #version 430 core
     #extension GL_ARB_separate_shader_objects : enable 
      
      layout (location = 0) in vec4 vertex;   
      layout (location = 1) in vec4 texcoord;   
      layout (location = 2) in vec4 normal;   
      
      uniform mat4 modelviewprojectionmatrix; 
      uniform mat4 modelviewmatrix; 
      uniform mat4 modelviewinversematrix; 
      uniform mat4 normalmatrix; 
      
      out vec4 fragnormal;  
      out vec4 fragtexcoord;
      out vec4 fragposition;
      out vec4 object_coords;
      
      void main(void) 
      { 
        fragtexcoord  = texcoord; 
        fragnormal    = normalmatrix * normal; 
        fragposition  = modelviewmatrix * vertex; 
        gl_Position   = modelviewprojectionmatrix * vertex; 
        object_coords = vertex;
    })"; 

    std::string fragmentshader_code = R"(
      #version 430 core
      #extension GL_ARB_separate_shader_objects : enable 
      
      in vec4 fragnormal;   
      in vec4 fragtexcoord; 
      in vec4 fragposition; 
      in vec4 object_coords; 

      #define HULLVERTEXMAP_SSBO_BINDING 0
      #define ATOMIC_COUNTER_BINDING 1
      #define ESTIMATE_COUNTER_BINDING 2

      uniform mat4 modelviewprojectionmatrix; 
      uniform mat4 modelviewmatrix; 
      uniform mat4 modelviewinversematrix; 

      uniform int window_width;
      uniform int window_height;

      uniform samplerBuffer obbdata;
      
      struct hull_vertex_entry {
        unsigned char id;
        unsigned char num_visible_vertices;
        unsigned char vertices[6];
      };

      layout(std430, binding = HULLVERTEXMAP_SSBO_BINDING) buffer hullvertexmap_ssbo {
        hull_vertex_entry gpucast_hvm[];
      };

      layout(std430, binding = ESTIMATE_COUNTER_BINDING) buffer estimate_ssbo {
        uint estimate_data[];
      };

      layout(binding = ATOMIC_COUNTER_BINDING, offset = 0) uniform atomic_uint fragment_counter;

      ///////////////////////////////////////////////////////////////////////////////
      void fetch_obb_data(in samplerBuffer obb_data, in int base_id, out vec4 obb_center, out mat4 obb_orientation, out mat4 obb_orientation_inverse, out vec4 obb_vertices[8])
      {
        obb_center = texelFetch(obb_data, base_id);

        obb_orientation = mat4(texelFetch(obb_data, base_id + 3),
                              texelFetch(obb_data, base_id + 4),
                              texelFetch(obb_data, base_id + 5),
                              texelFetch(obb_data, base_id + 6));

        obb_orientation_inverse = mat4(texelFetch(obb_data, base_id + 7),
                                      texelFetch(obb_data, base_id + 8),
                                      texelFetch(obb_data, base_id + 9),
                                      texelFetch(obb_data, base_id + 10));

        for (int i = 0; i != 8; ++i) {
          obb_vertices[i] = texelFetch(obb_data, base_id + 11 + i);
        }
      }

      ///////////////////////////////////////////////////////////////////////////////
      float calculate_obb_area(in mat4            modelview_projection,
                                in mat4           modelview_inverse,
                                in samplerBuffer  obb_data,
                                in int            obb_base_index,
                                in bool           clamp_to_screen,
                                in uint           frag )
      {
        vec4 obb_center;
        mat4 obb_orientation;
        mat4 obb_orientation_inverse;
        vec4 bbox[8];

        fetch_obb_data(obb_data, obb_base_index, obb_center, obb_orientation, obb_orientation_inverse, bbox);

        // transform eye to obb space
        vec4 eye_object_space = modelview_inverse * vec4(0.0, 0.0, 0.0, 1.0);
        vec4 eye_obb_space = obb_orientation_inverse * vec4(eye_object_space.xyz - obb_center.xyz, 1.0);

        // identify in which quadrant the eye is located
        float sum = 0.0;

        int pos = (int(eye_obb_space.x < bbox[0].x))        //  1 = left   |  compute 6-bit
                + (int(eye_obb_space.x > bbox[6].x) << 1)   //  2 = right  |        code to
                + (int(eye_obb_space.y < bbox[0].y) << 2)   //  4 = bottom |   classify eye
                + (int(eye_obb_space.y > bbox[6].y) << 3)   //  8 = top    |with respect to
                + (int(eye_obb_space.z < bbox[0].z) << 4)   // 16 = front  | the 6 defining
                + (int(eye_obb_space.z > bbox[6].z) << 5);  // 32 = back   |         planes

        if (frag < 1) {          
          atomicMax(estimate_data[2], pos);
        }

        // look up according number of visible vertices
        int n_visible_vertices = int(gpucast_hvm[pos].num_visible_vertices);
        if (n_visible_vertices == 0) {
          return 0.0;
        }

        if (frag < 1) {
          atomicMax(estimate_data[1], n_visible_vertices);
        }

          // project all obb vertices to screen coordinates
        vec2 dst[6];
        for (int i = 0; i != n_visible_vertices; ++i) {
          uint index = gpucast_hvm[pos].vertices[i];
          vec4 corner_screenspace = modelview_projection * (obb_orientation * bbox[int(index)] + vec4(obb_center.xyz, 0.0));
          corner_screenspace /= corner_screenspace.w;

          // if clamped parts at the border appear to coarsly tesselated
          if (clamp_to_screen) {
            dst[i] = clamp(corner_screenspace.xy, vec2(-1.0), vec2(1.0));
          } else {
            dst[i] = corner_screenspace.xy;
          }
        }

        // accumulate area of visible vertices' polygon
        for (int i = 0; i < n_visible_vertices; i++) {
          sum += (dst[i].x - dst[(i + 1) % n_visible_vertices].x) * (dst[i].y + dst[(i + 1) % n_visible_vertices].y);
        }

        // return area
        return abs(sum) / 8.0; // this differs from original, but testet with extra application
      }

      layout (location = 0) out vec4 color; 
      
      void main(void) 
      { 
        vec4 obb_center;
        mat4 obb_orientation;
        mat4 obb_orientation_inverse;
        vec4 bbox[8];

        fetch_obb_data(obbdata, 0, obb_center, obb_orientation, obb_orientation_inverse, bbox);

        vec4 eye_object_space = modelviewinversematrix * vec4(0.0, 0.0, 0.0, 1.0);
        
        eye_object_space = vec4(eye_object_space.xyz - obb_center.xyz, 1.0);
        
        vec4 eye_obb_space = obb_orientation_inverse * eye_object_space;

        color = normalize(obb_orientation_inverse * vec4(object_coords.xyz - obb_center.xyz, 1.0));  
         
        uint nfrag = atomicCounterIncrement(fragment_counter);

        float estimate = calculate_obb_area(modelviewprojectionmatrix, 
                                            modelviewinversematrix,
                                            obbdata,
                                            0,
                                            true, nfrag);

        uint estimate_in_pixel = uint(window_height * window_width * estimate); 

        if (nfrag < 1) {
          uint max_estimate = atomicMax(estimate_data[0], estimate_in_pixel);
        }
      })";
  
    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);
 
    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());
    
    vs.compile();
    fs.compile();

    _program.add(&fs);
    _program.add(&vs);

    std::cout << "vertex shader log : " << vs.log() << std::endl;
    std::cout << "fragment shader log : " << fs.log() << std::endl;

    _program.link();   
  }


  ///////////////////////////////////////////////////////////////////////////////
  void fetch_obb_data(std::vector<gpucast::math::vec4f> const& obb_data, int base_id, gpucast::math::vec4f& obb_center, gpucast::math::matrix4f& obb_orientation, gpucast::math::matrix4f& obb_orientation_inverse, gpucast::math::vec4f* obb_vertices)
  {
    typedef gpucast::math::matrix4f mat4;
    typedef gpucast::math::vec4f vec4;

    auto texelFetch = [] (std::vector<gpucast::math::vec4f> const& v, unsigned id) {return v[id]; };

    obb_center = texelFetch(obb_data, base_id);

    obb_orientation[0] = texelFetch(obb_data, base_id + 3)[0];
    obb_orientation[1] = texelFetch(obb_data, base_id + 3)[1];
    obb_orientation[2] = texelFetch(obb_data, base_id + 3)[2];
    obb_orientation[3] = texelFetch(obb_data, base_id + 3)[3];

    obb_orientation[4] = texelFetch(obb_data, base_id + 4)[0];
    obb_orientation[5] = texelFetch(obb_data, base_id + 4)[1];
    obb_orientation[6] = texelFetch(obb_data, base_id + 4)[2];
    obb_orientation[7] = texelFetch(obb_data, base_id + 4)[3];

    obb_orientation[8] = texelFetch(obb_data, base_id  + 5)[0];
    obb_orientation[9] = texelFetch(obb_data, base_id  + 5)[1];
    obb_orientation[10] = texelFetch(obb_data, base_id + 5)[2];
    obb_orientation[11] = texelFetch(obb_data, base_id + 5)[3];

    obb_orientation[12] = texelFetch(obb_data, base_id + 6)[0];
    obb_orientation[13] = texelFetch(obb_data, base_id + 6)[1];
    obb_orientation[14] = texelFetch(obb_data, base_id + 6)[2];
    obb_orientation[15] = texelFetch(obb_data, base_id + 6)[3];

    obb_orientation_inverse[0] = texelFetch(obb_data, base_id + 7)[0];
    obb_orientation_inverse[1] = texelFetch(obb_data, base_id + 7)[1];
    obb_orientation_inverse[2] = texelFetch(obb_data, base_id + 7)[2];
    obb_orientation_inverse[3] = texelFetch(obb_data, base_id + 7)[3];

    obb_orientation_inverse[4] = texelFetch(obb_data, base_id + 8)[0];
    obb_orientation_inverse[5] = texelFetch(obb_data, base_id + 8)[1];
    obb_orientation_inverse[6] = texelFetch(obb_data, base_id + 8)[2];
    obb_orientation_inverse[7] = texelFetch(obb_data, base_id + 8)[3];

    obb_orientation_inverse[8] = texelFetch(obb_data, base_id +  9)[0];
    obb_orientation_inverse[9] = texelFetch(obb_data, base_id +  9)[1];
    obb_orientation_inverse[10] = texelFetch(obb_data, base_id + 9)[2];
    obb_orientation_inverse[11] = texelFetch(obb_data, base_id + 9)[3];

    obb_orientation_inverse[12] = texelFetch(obb_data, base_id + 10)[0];
    obb_orientation_inverse[13] = texelFetch(obb_data, base_id + 10)[1];
    obb_orientation_inverse[14] = texelFetch(obb_data, base_id + 10)[2];
    obb_orientation_inverse[15] = texelFetch(obb_data, base_id + 10)[3];

    for (int i = 0; i != 8; ++i) {
      obb_vertices[i] = texelFetch(obb_data, base_id + 11 + i);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  float calculate_obb_area(gpucast::math::matrix4f const& modelview_projection,
                           gpucast::math::matrix4f const& modelview_inverse,
                           std::vector<gpucast::math::vec4f> const& obb_data,
                           int obb_base_index,
                           bool clamp_to_screen)
  {
    typedef gpucast::math::matrix4f mat4;
    typedef gpucast::math::vec4f vec4;
    typedef gpucast::math::vec2f vec2;

    vec4 obb_center;
    mat4 obb_orientation;
    mat4 obb_orientation_inverse;
    vec4 bbox[8];

    fetch_obb_data(obb_data, obb_base_index, obb_center, obb_orientation, obb_orientation_inverse, bbox);

    // transform eye to obb space
    vec4 eye_object_space = modelview_inverse * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 eye_object_space_translated = eye_object_space - obb_center;
    eye_object_space_translated[3] = 1.0;

    vec4 eye_obb_space = obb_orientation_inverse * eye_object_space_translated;

    // identify in which quadrant the eye is located
    float sum = 0.0;
    int pos = (int(eye_obb_space[0] < bbox[0][0]))        //  1 = left   |  compute 6-bit
            + (int(eye_obb_space[0] > bbox[6][0]) << 1)   //  2 = right  |        code to
            + (int(eye_obb_space[1] < bbox[0][1]) << 2)   //  4 = bottom |   classify eye
            + (int(eye_obb_space[1] > bbox[6][1]) << 3)   //  8 = top    |with respect to
            + (int(eye_obb_space[2] < bbox[0][2]) << 4)   // 16 = front  | the 6 defining
            + (int(eye_obb_space[2] > bbox[6][2]) << 5);  // 32 = back   |         planes

    // look up according number of visible vertices
    gpucast::gl::hullvertexmap hvm;
    int n_visible_vertices = int(hvm.data[pos].num_visible_vertices);
    if (n_visible_vertices == 0) {
      return 0.0;
    }

    // project all obb vertices to screen coordinates
    vec2 dst[6];
    for (int i = 0; i != n_visible_vertices; ++i) {
      int index = hvm.data[pos].visible_vertex[i];
      vec4 corner_rotated = obb_orientation * bbox[index];
      vec4 corner_translated = corner_rotated + obb_center;
      corner_translated[3] = 1.0;

      vec4 corner_screenspace = modelview_projection * corner_translated;
      corner_screenspace /= corner_screenspace[3];

      // if clamped parts at the border appear to coarsly tesselated
      if (clamp_to_screen) {
        dst[i] = vec2(clamp(corner_screenspace[0], -1.0f, 1.0f), clamp(corner_screenspace[1], -1.0f, 1.0f));
      }
      else {
        dst[i] = vec2(corner_screenspace[0], corner_screenspace[1]);
      }
    }

    // accumulate area of visible vertices' polygon
    for (int i = 0; i < n_visible_vertices; i++) {
      sum += (dst[i][0] - dst[(i + 1) % n_visible_vertices][0]) * (dst[i][1] + dst[(i + 1) % n_visible_vertices][1]);
    }

    // return area
    return abs(sum) / 8.0; // this differs from original, because in space [-1, 1]
  }


  /////////////////////////////////////////////////////////////////////////////
  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glFrontFace(GL_CW);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, float(window_width) / float(window_height), 1.0f, 1000.0f);
    gpucast::math::matrix4f mv   = view * model;
    gpucast::math::matrix4f mvi =  gpucast::math::inverse(mv);
    gpucast::math::matrix4f mvp  = proj * mv;
    gpucast::math::matrix4f nm   = mv.normalmatrix();

    if (!_counter) {
      _counter = std::make_shared<gpucast::gl::atomicbuffer>(2 * sizeof(unsigned int), GL_DYNAMIC_COPY);
    }

    clear_counter();

    // bind atomic counter buffer and draw
    _counter->bind_buffer_base(ATOMIC_COUNTER_BINDING);

    _program.begin();

    _program.set_texturebuffer("obbdata", _obb, OBB_TEXTURE_UNIT);
    _program.set_shaderstoragebuffer("hullvertexmap_ssbo", _hvm, HULLVERTEXMAP_SSBO_BINDING);
    _program.set_shaderstoragebuffer("estimate_ssbo", _estimate, ESTIMATE_COUNTER_BINDING);
      
    _program.set_uniform1i("window_width", window_width);
    _program.set_uniform1i("window_height", window_height);

    _program.set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program.set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program.set_uniform_matrix4fv("modelviewinversematrix", 1, false, &mvi[0]);
    _program.set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    _cube.draw();

    _program.end();

    print_counter();
  }


  void clear_counter()
  {
    // initialize buffer with 0
    _counter->bind();
    unsigned* mapped_mem_write = (unsigned*)_counter->map_range(0, sizeof(unsigned), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    *mapped_mem_write = 0;
    _counter->unmap();

    _estimate.bind();
    mapped_mem_write = (unsigned*)_estimate.map_range(0, sizeof(unsigned), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    for (auto i = 0; i != 100; ++i) {
      *(mapped_mem_write + i) = 0;
    }
    _estimate.unmap();
  }


  void print_counter() 
  {
    _counter->bind();
    unsigned* mapped_mem_read = (unsigned*)_counter->map_range(0, sizeof(unsigned), GL_MAP_READ_BIT);

    if (_framecount % 100 == 0) {
      std::cout << "#Fragments (atomic counter): " << *mapped_mem_read << std::endl;
    }

    _counter->unmap();
    _counter->unbind();

    _estimate.bind();
    mapped_mem_read = (unsigned*)_estimate.map_range(0, sizeof(unsigned), GL_MAP_READ_BIT);
    if (_framecount++ % 100 == 0) {
      std::cout << "OBB case : " << *(mapped_mem_read + 2) << std::endl;
      std::cout << "#Visible vertices : " << *(mapped_mem_read + 1) << std::endl;
      std::cout << "#Estimated fragments GPU : " << *(mapped_mem_read) << std::endl;

      gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f);

      gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
        _trackball->rotation();

      gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, float(window_width) / float(window_height), 1.0f, 1000.0f);
      gpucast::math::matrix4f mv = view * model;
      gpucast::math::matrix4f mvi = gpucast::math::inverse(mv);
      gpucast::math::matrix4f mvp = proj * mv;
      std::cout << "#Estimated fragments CPU : " << window_width * window_height * calculate_obb_area(mvp, mvi, _obbdata, 0, false) << std::endl;
    }
    _estimate.unmap();
    _estimate.unbind();
  }

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


public :

  unsigned                                   _framecount;
  gpucast::gl::program                       _program;
  gpucast::gl::cube                          _cube;

  gpucast::gl::texturebuffer                 _obb;
  std::vector<gpucast::math::vec4f>          _obbdata;

  gpucast::gl::shaderstoragebuffer           _hvm;
  gpucast::gl::shaderstoragebuffer           _estimate;

  std::shared_ptr<gpucast::gl::atomicbuffer> _counter;
  std::shared_ptr<gpucast::gl::trackball>    _trackball;
};


int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, window_width, window_height, 0, 0, 4, 4, false);

  glewExperimental = true;
  glewInit();

  application app;
  app.run();

  return 0;
}
