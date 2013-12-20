/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : simple_opencl.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// system includes
#include <iostream>
#include <fstream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <CL/cl.hpp>

#include <boost/bind.hpp>

// local includes
#include <glpp/glut/window.hpp>
#include <glpp/util/camera.hpp>
#include <glpp/util/timer.hpp>
#include <glpp/math/vec4.hpp>

#include <tml/parametric/beziervolume.hpp>

// opencl header
#include <boost/foreach.hpp>



class application
{
public :

  application()
    : _camera   (),
      _samples  (256*256*256)
  {
    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);

    initcl();
    init_volume();
  }

  void init_volume() 
  {
    glpp::vec4f data[27];

    data[0] = glpp::vec4f(0.0, 0.0, 0.0, 1.0);
    data[1] = glpp::vec4f(1.0, 0.0, 0.0, 1.0);
    data[2] = glpp::vec4f(2.0, 0.0, 0.0, 1.0);
    data[0] = glpp::vec4f(0.0, 1.0, 0.0, 1.0);
    data[1] = glpp::vec4f(1.0, 1.0, 0.0, 1.0);
    data[2] = glpp::vec4f(2.0, 1.0, 0.0, 1.0);
    data[0] = glpp::vec4f(0.0, 2.0, 0.0, 1.0);
    data[1] = glpp::vec4f(1.0, 2.0, 0.0, 1.0);
    data[2] = glpp::vec4f(2.0, 2.0, 0.0, 1.0);

    data[0] = glpp::vec4f(0.0, 0.0, 1.0, 1.0);
    data[1] = glpp::vec4f(1.0, 0.0, 1.0, 1.0);
    data[2] = glpp::vec4f(2.0, 0.0, 1.0, 1.0);
    data[0] = glpp::vec4f(0.0, 1.0, 1.0, 1.0);
    data[1] = glpp::vec4f(1.0, 1.0, 1.0, 1.0);
    data[2] = glpp::vec4f(2.0, 1.0, 1.0, 1.0);
    data[0] = glpp::vec4f(0.0, 2.0, 1.0, 1.0);
    data[1] = glpp::vec4f(1.0, 2.0, 1.0, 1.0);
    data[2] = glpp::vec4f(2.0, 2.0, 1.0, 1.0);

    data[0] = glpp::vec4f(0.0, 0.0, 2.0, 1.0);
    data[1] = glpp::vec4f(1.0, 0.0, 2.0, 1.0);
    data[2] = glpp::vec4f(2.0, 0.0, 2.0, 1.0);
    data[0] = glpp::vec4f(0.0, 1.0, 2.0, 1.0);
    data[1] = glpp::vec4f(1.0, 1.0, 2.0, 1.0);
    data[2] = glpp::vec4f(2.0, 1.0, 2.0, 1.0);
    data[0] = glpp::vec4f(0.0, 2.0, 2.0, 1.0);
    data[1] = glpp::vec4f(1.0, 2.0, 2.0, 1.0);
    data[2] = glpp::vec4f(2.0, 2.0, 2.0, 1.0);
  }

  void initcl()
  {
    cl_int err = CL_SUCCESS;

    // Get list of platforms (things that can execute OpenCL on this host), get a "context" on the first executor.
    std::vector<cl::Platform> available_platforms;

    if (cl::Platform::get(&available_platforms) != CL_SUCCESS){
      std::cerr << "Failed to retrieve platform.\n";
    }

    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, 
                                        (cl_context_properties)(available_platforms[0])(), 
                                        0 };

    cl::Context context (CL_DEVICE_TYPE_GPU, cprops);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
 
    // Create a command queue and use the first device
    cl::CommandQueue queue (context, devices[0], 0, &err);
    if ( err != CL_SUCCESS ) 
    {
      std::cerr << "Error creating Command queue: " << err << std::endl;
    } else {
      std::cout << "Creating Command queue ok." << std::endl;
    }
 
    std::string info;
    devices[0].getInfo(CL_DEVICE_NAME, &info);
    std::cout << "Device Name : " << info << std::endl;

    devices[0].getInfo(CL_DEVICE_VENDOR, &info);
    std::cout << "Device Vendor : " << info << std::endl;

    devices[0].getInfo(CL_DRIVER_VERSION, &info);
    std::cout << "Device Driver Version : " << info << std::endl;

    devices[0].getInfo(CL_DEVICE_VERSION, &info);
    std::cout << "Device Device Version : " << info << std::endl;

    // Read source file
    std::ifstream sourceFile("../evaluate_curve.cl");
    std::string sourceCode  ( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
 
    // Make program of the source code in the context
    cl::Program program (context, source);
 
    // Build program for these specific devices
    program.build(devices);
    std::string buildlog;
    program.getBuildInfo( devices[0], (cl_program_build_info)CL_PROGRAM_BUILD_LOG, &buildlog );

    std::cout << "log: " << buildlog << std::endl;

    // Make kernel
    cl::Kernel kernel(program, "evaluate_curve", &err);
    if ( err != CL_SUCCESS ) 
    {
      std::cerr << "Error creating kernel: " << err << std::endl;
    } else {
      std::cout << "Creating kernel ok." << std::endl;
    }
    
    std::vector<glpp::vec4f> points;
    points.push_back(glpp::vec4f(1.0, 0.0, 0.0, 1.0));
    points.push_back(glpp::vec4f(3.0, 4.0,-3.0, 1.0));
    points.push_back(glpp::vec4f(2.0, 2.0, 7.0, 1.0));
    points.push_back(glpp::vec4f(0.0, 4.0, 0.0, 1.0));

    int const order = points.size();

    // Create memory buffers
    cl::Buffer bufferA (context, CL_MEM_READ_ONLY,  order * sizeof(glpp::vec4f), 0, &err);
    if ( err != CL_SUCCESS ) {
      std::cerr << "Error allocating CL memory : " << err << std::endl;
    } else {
      std::cout << "Allocating memory ok." << std::endl;
    }

    cl::Buffer bufferB (context, CL_MEM_WRITE_ONLY, _samples * sizeof(glpp::vec4f), 0, &err);
    if ( err != CL_SUCCESS ) {
      std::cerr << "Error allocating CL memory : " << err << std::endl;
    } else {
      std::cout << "Allocating memory ok." << std::endl;
    }

    std::vector<glpp::vec4f> samples(_samples);
    std::vector<glpp::vec4f> result (_samples);

    // Copy lists A and B to the memory buffers
    queue.enqueueWriteBuffer (bufferA, CL_TRUE, 0, order * sizeof(glpp::vec4f), &points[0]);
    queue.enqueueWriteBuffer (bufferB, CL_TRUE, 0, _samples * sizeof(glpp::vec4f), &samples[0]);

    glpp::timer t;
    glpp::timer t2;
    t.start();
    
    // Set arguments to kernel
    kernel.setArg(0, order);
    kernel.setArg(1, bufferA);
    kernel.setArg(2, bufferB);
    
    int const runs = 100;
    for (int i = 0; i != runs; ++i)
    {
      // Run the kernel on specific ND range
      cl::NDRange global (_samples);
      cl::NDRange local  (32);
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    }

    queue.flush();
    queue.enqueueBarrier();
    queue.finish();

    t.stop();
    std::cout << "Computation time for " << runs * _samples << " evaluations : " << t.result() << std::endl; 

    t2.start();
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, _samples * sizeof(glpp::vec4f), &result[0]);

    //std::copy(result.begin(), result.end(), std::ostream_iterator<glpp::vec4f>(std::cout, "\n"));
    
    t2.stop();
    std::cout << "Including readback : " << t2.result() << std::endl; 

    std::cout << "C(0.0) = " << result.front() << std::endl;
    std::cout << "C(0.5) = " << result[_samples/2] << std::endl;
    std::cout << "C(1.0) = " << result.back() << std::endl;
  }
  

  void run()
  {
    glpp::glutwindow::instance().run();
  }

  void draw()
  {

  }

private :

  glpp::camera                        _camera;
  unsigned                            _samples;

  // some opencl memory
  cl_mem                              _controlpoints;
  cl_mem                              _points;
};


int main(int argc, char** argv)
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 3, 3, false);
  glewInit();

  application app;
  app.run();

  return 0;
}
