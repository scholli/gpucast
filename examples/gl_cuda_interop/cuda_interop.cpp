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

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/texture2d.hpp>

#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/primitives/plane.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_helper.hpp>
#include <cuda_interop.hpp>

class application
{
public:

  /////////////////////////////////////////////////////////////////////////////
  application(int w, int h)
    : _program(),
      _plane(0, -1, 1),
      _trackball(new gpucast::gl::trackball),
      _texture(),
      _linear_sampler(),
      _window_width(w),
      _window_height(h)
  {
    init_shader();
    
    resize_texture(_texture_width, _texture_height);

    _linear_sampler.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _linear_sampler.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));

    glEnable(GL_DEPTH_TEST);
  }

  /////////////////////////////////////////////////////////////////////////////
  ~application()
  {
    if (_cuda_initialized && _cuda_texture_handle) {
      unregister_resource(&_cuda_texture_handle);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  bool init_cuda()
  {
    int devID = 0;
    devID = gpuDeviceInit(devID);

    if (devID < 0)
    {
      std::cerr << "gpuDeviceInit() failed. " << std::endl;
      return false;
    } 

    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
      
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    std::cout << "GPU Device " << devID << ": " << deviceProp.name << "with compute capability " << deviceProp.major << "." << deviceProp.minor << "\n";
    
    return true;
  }

  /////////////////////////////////////////////////////////////////////////////
  void resize_texture(int width, int height) 
  {
    if (_cuda_texture_handle) {
      unregister_resource(&_cuda_texture_handle);
    }
    _texture.resize(width, height, GL_RGBA32F);

    _texture_width = width;
    _texture_height = height;

    randomize_texture();

    if (_cuda_initialized) {
      register_image(&_cuda_texture_handle, _texture.id(), _texture.target(), cudaGraphicsRegisterFlagsSurfaceLoadStore);
    }
  } 

  /////////////////////////////////////////////////////////////////////////////
  void randomize_texture() 
  {
    std::vector<float> random(_texture_width * _texture_height * 4);
    std::generate(random.begin(), random.end(), []() -> float { return float(std::rand()) / RAND_MAX; });
    _texture.texsubimage(0, 0, 0, _texture_width, _texture_height, GL_RGBA, GL_FLOAT, &random[0]);
  }
  /////////////////////////////////////////////////////////////////////////////
  void register_image(cudaGraphicsResource_t* resource, GLuint image, GLenum target, unsigned int flags)
  {
    if (!(*resource)) {
      std::cout << "Register GL Resources for Image ..." << std::endl;
      checkCudaErrors(cudaGraphicsGLRegisterImage(resource, image, target, flags));
    }
    else {
      std::cerr << " CudaRessource already registered ... " << std::endl;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  void unregister_resource(cudaGraphicsResource_t* resource)
  {
    if (*resource)
    {
      checkCudaErrors(cudaGraphicsUnregisterResource(*resource));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void init_shader()
  {
    std::string vertexshader_code = R"(
      #version 440 core
 
      layout (location = 0) in vec4 vertex;   
      layout (location = 1) in vec4 texcoord;   
 
      uniform mat4 modelviewprojectionmatrix; 

      out vec4 fragtexcoord;

      void main(void) 
      { 
        fragtexcoord = texcoord; 
        gl_Position  = modelviewprojectionmatrix * vertex; 
    })";

    std::string fragmentshader_code = R"(
      #version 440 core

      in vec4 fragtexcoord; 

      uniform sampler2D cuda_result;
      layout (location = 0) out vec4 color; 
      
      void main(void) 
      { 
        
        color = texture(cuda_result, fragtexcoord.xy);
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

  /////////////////////////////////////////////////////////////////////////////
  void draw()
  {
    // CPU inititialization of random texture
    if (!_cuda_initialized) {
      _cuda_initialized = init_cuda();
      resize_texture(_texture_width, _texture_height);
    }
    else {
      randomize_texture();
    }

    // here comes CUDA
    invoke_square_kernel(_texture_width, _texture_height, _cuda_texture_handle);
    checkCudaErrors(cudaDeviceSynchronize());

    // here comes openGL
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f,
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
      _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, float(_window_width)/_window_height, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv = view * model;
    gpucast::math::matrix4f mvp = proj * mv;
    
    _program.begin();

    _program.set_texture2d("cuda_result", _texture, 0);
    _linear_sampler.bind(0);
    _program.set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);

    _plane.draw();

    _program.end();
  }

  /////////////////////////////////////////////////////////////////////////////
  void run()
  {
    gpucast::gl::glutwindow::instance().run();
  }


public:

  unsigned                                _window_width;
  unsigned                                _window_height;

  unsigned                                _texture_width = 256;
  unsigned                                _texture_height = 256;

  bool                                    _cuda_initialized = false;

  gpucast::gl::texture2d                  _texture;
  cudaGraphicsResource_t                  _cuda_texture_handle = nullptr;
  gpucast::gl::sampler                    _linear_sampler;

  gpucast::gl::program                    _program;
  gpucast::gl::plane                      _plane;

  std::shared_ptr<gpucast::gl::trackball> _trackball;
};


int main(int argc, char** argv)
{
  static unsigned initial_window_width = 1920;
  static unsigned initial_window_height = 1080;

  gpucast::gl::glutwindow::init(argc, argv, initial_window_width, initial_window_height, 0, 0, 4, 4, false);

  glewExperimental = true;
  glewInit();

  application app(initial_window_width, initial_window_height);
  app.run();

  return 0;
}
