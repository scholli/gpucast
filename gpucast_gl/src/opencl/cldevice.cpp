/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : cldevice.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast_gl/opencl/cldevice.hpp"

// system
#include <stdexcept>
#include <cstring>

#include <gpucast_gl/opencl/clcontext.hpp>


namespace gpucast { namespace gl {


///////////////////////////////////////////////////////////////////////////////
cldevice::cldevice()
{}


///////////////////////////////////////////////////////////////////////////////
cldevice::~cldevice()
{}



///////////////////////////////////////////////////////////////////////////////
boost::shared_ptr<clcontext>
cldevice::create_context () const
{
  boost::shared_ptr<clcontext> context (new clcontext);
  cl_int err;

  context->_id     = clCreateContext      ( 0, 1, &_id, NULL, NULL, &err);
  if ( err != CL_SUCCESS ) {
    throw std::runtime_error("clCreateContext() failed to create context");
  }

  context->_queue  = clCreateCommandQueue ( context->_id, _id, 0, &err);
  if ( err != CL_SUCCESS ) {
    throw std::runtime_error("clCreateCommandQueue() failed to create command queue");
  }

  return context;
}


///////////////////////////////////////////////////////////////////////////////
void
cldevice::print ( std::ostream& os) const
{
  char _id_string[1024];
  bool nv_device_attibute_query = false;

  // CL_DEVICE_NAME
  clGetDeviceInfo(_id, CL_DEVICE_NAME, sizeof(_id_string), &_id_string, NULL);
  os << "  CL_DEVICE_NAME: " << _id_string << std::endl;

  // CL_DEVICE_VENDOR
  clGetDeviceInfo(_id, CL_DEVICE_VENDOR, sizeof(_id_string), &_id_string, NULL);
  os << "  CL_DEVICE_VENDOR: " << _id_string << std::endl;

  // CL_DRIVER_VERSION
  clGetDeviceInfo(_id, CL_DRIVER_VERSION, sizeof(_id_string), &_id_string, NULL);
  os << "  CL_DRIVER_VERSION: " << _id_string << std::endl;

  // CL_DEVICE_VERSION
  clGetDeviceInfo(_id, CL_DEVICE_VERSION, sizeof(_id_string), &_id_string, NULL);
  os << "  CL_DEVICE_VERSION: " << _id_string << std::endl;

  // CL_DEVICE_OPENCL_C_VERSION (if CL_DEVICE_VERSION version > 1.0)
  if ( strncmp("OpenCL 1.0", _id_string, 10) != 0 )
  {
      // This code is unused for _ids reporting OpenCL 1.0, but a def is needed anyway to allow compilation using v 1.0 headers
      // This constant isn't #defined in 1.0
      #ifndef CL_DEVICE_OPENCL_C_VERSION
          #define CL_DEVICE_OPENCL_C_VERSION 0x103D
      #endif

      clGetDeviceInfo(_id, CL_DEVICE_OPENCL_C_VERSION, sizeof(_id_string), &_id_string, NULL);
      os << "  CL_DEVICE_OPENCL_C_VERSION: " << _id_string << std::endl;
  }

  // CL_DEVICE_TYPE
  cl_device_type type;
  clGetDeviceInfo(_id, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  if( type & CL_DEVICE_TYPE_CPU )
      os << "  CL_DEVICE_TYPE: " << "CL_DEVICE_TYPE_CPU" << std::endl;
  if( type & CL_DEVICE_TYPE_GPU )
      os << "  CL_DEVICE_TYPE: " << "CL_DEVICE_TYPE_GPU" << std::endl;
  if( type & CL_DEVICE_TYPE_ACCELERATOR )
      os << "  CL_DEVICE_TYPE: " << "CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
  if( type & CL_DEVICE_TYPE_DEFAULT )
      os << "  CL_DEVICE_TYPE: " << "CL_DEVICE_TYPE_DEFAULT" << std::endl;

  // CL_DEVICE_MAX_COMPUTE_UNITS
  cl_uint compute_units;
  clGetDeviceInfo(_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
  os << "  CL_DEVICE_MAX_COMPUTE_UNITS: " << compute_units << std::endl;

  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  size_t workitem_dims;
  clGetDeviceInfo(_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
  os << "  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << workitem_dims << std::endl;

  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  size_t workitem_size[3];
  clGetDeviceInfo(_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
  os << "  CL_DEVICE_MAX_WORK_ITEM_SIZES: " << workitem_size[0] << " " << workitem_size[1] << " " << workitem_size[2] << std::endl;

  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  size_t workgroup_size;
  clGetDeviceInfo(_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
  os << "  CL_DEVICE_MAX_WORK_GROUP_SIZE: " << workgroup_size << std::endl;


  // CL_DEVICE_MAX_CLOCK_FREQUENCY
  cl_uint clock_frequency;
  clGetDeviceInfo(_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
  os << "  CL_DEVICE_MAX_CLOCK_FREQUENCY: " << clock_frequency << std::endl;

  // CL_DEVICE_ADDRESS_BITS
  cl_uint addr_bits;
  clGetDeviceInfo(_id, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
  os << "  CL_DEVICE_MAX_CLOCK_FREQUENCY: " << addr_bits << std::endl;

  // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  cl_ulong max_mem_alloc_size;
  clGetDeviceInfo(_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
  os << "  CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << (unsigned int)(max_mem_alloc_size / (1024 * 1024)) << std::endl;

  // CL_DEVICE_GLOBAL_MEM_SIZE
  cl_ulong mem_size;
  clGetDeviceInfo(_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  os << "  CL_DEVICE_GLOBAL_MEM_SIZE: " << (unsigned int)(mem_size / (1024 * 1024)) << std::endl;

  // CL_DEVICE_ERROR_CORRECTION_SUPPORT
  cl_bool error_correction_support;
  clGetDeviceInfo(_id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
  os << "  CL_DEVICE_ERROR_CORRECTION_SUPPORT: " << (error_correction_support == CL_TRUE ? "yes" : "no") << std::endl;

  // CL_DEVICE_LOCAL_MEM_TYPE
  cl_device_local_mem_type local_mem_type;
  clGetDeviceInfo(_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
  os << "  CL_DEVICE_LOCAL_MEM_TYPE: " << (local_mem_type == 1 ? "local" : "global") << std::endl;

  // CL_DEVICE_LOCAL_MEM_SIZE
  clGetDeviceInfo(_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  os << "  CL_DEVICE_LOCAL_MEM_SIZE: " << (unsigned int)(mem_size / 1024) << std::endl;

  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  clGetDeviceInfo(_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
  os << "  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << (unsigned int)(mem_size / 1024) << std::endl;

  // CL_DEVICE_QUEUE_PROPERTIES
  cl_command_queue_properties queue_properties;
  clGetDeviceInfo(_id, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
  if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
    os << "  CL_DEVICE_QUEUE_PROPERTIES: " << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE" << std::endl;
  if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
    os << "  CL_DEVICE_QUEUE_PROPERTIES: " << "CL_QUEUE_PROFILING_ENABLE" << std::endl;

  // CL_DEVICE_IMAGE_SUPPORT
  cl_bool image_support;
  clGetDeviceInfo(_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
  os << "  CL_DEVICE_IMAGE_SUPPORT: " << image_support << std::endl;

  // CL_DEVICE_MAX_READ_IMAGE_ARGS
  cl_uint max_read_image_args;
  clGetDeviceInfo(_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
  os << "  CL_DEVICE_MAX_READ_IMAGE_ARGS: " << max_read_image_args << std::endl;

  // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
  cl_uint max_write_image_args;
  clGetDeviceInfo(_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
  os << "  CL_DEVICE_MAX_WRITE_IMAGE_ARGS: " << max_write_image_args << std::endl;

  // CL_DEVICE_SINGLE_FP_CONFIG
  /*cl__id_fp_config fp_config;
  clGetDeviceInfo(_id, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl__id_fp_config), &fp_config, NULL);
  shrLogEx(iLogMode, 0, "  CL_DEVICE_SINGLE_FP_CONFIG:\t\t%s%s%s%s%s%s\n",
      fp_config & CL_FP_DENORM ? "denorms " : "",
      fp_config & CL_FP_INF_NAN ? "INF-quietNaNs " : "",
      fp_config & CL_FP_ROUND_TO_NEAREST ? "round-to-nearest " : "",
      fp_config & CL_FP_ROUND_TO_ZERO ? "round-to-zero " : "",
      fp_config & CL_FP_ROUND_TO_INF ? "round-to-inf " : "",
      fp_config & CL_FP_FMA ? "fma " : "");
  */
  // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
  size_t szMaxDims[5];
  os << "\n  CL_DEVICE_IMAGE <dim>" << std::endl;
  clGetDeviceInfo(_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
  os << "  CL_DEVICE_IMAGE2D_MAX_WIDTH: " << szMaxDims[0] << std::endl;
  clGetDeviceInfo(_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
  os << "  CL_DEVICE_IMAGE2D_MAX_HEIGHT: " << szMaxDims[1] << std::endl;
  clGetDeviceInfo(_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
  os << "  CL_DEVICE_IMAGE3D_MAX_WIDTH: " << szMaxDims[2] << std::endl;
  clGetDeviceInfo(_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
  os << "  CL_DEVICE_IMAGE3D_MAX_HEIGHT: " << szMaxDims[3] << std::endl;
  clGetDeviceInfo(_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
  os << "  CL_DEVICE_IMAGE3D_MAX_DEPTH: " << szMaxDims[4] << std::endl;

  // CL_DEVICE_EXTENSIONS: get _id extensions, and if any then parse & log the string onto separate lines
  clGetDeviceInfo(_id, CL_DEVICE_EXTENSIONS, sizeof(_id_string), &_id_string, NULL);
  if (_id_string != 0)
  {
    os << "\n  CL_DEVICE_EXTENSIONS:" << std::endl;
    std::string stdDevString;
    stdDevString = std::string(_id_string);
    size_t szOldPos = 0;
    size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
    while (szSpacePos != stdDevString.npos)
    {
      if( strcmp("cl_nv__id_attribute_query", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 )
          nv_device_attibute_query = true;

      if (szOldPos > 0)
      {
        //shrLogEx(iLogMode, 0, "\t\t");
      }
      os << stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str() << std::endl;

      do {
        szOldPos = szSpacePos + 1;
        szSpacePos = stdDevString.find(' ', szOldPos);
      } while (szSpacePos == szOldPos);
    }
    os << std::endl;
  }
  else
  {
    os << "  CL_DEVICE_EXTENSIONS: None\n" << std::endl;
  }

  if(nv_device_attibute_query)
  {
      cl_uint compute_capability_major, compute_capability_minor;
      clGetDeviceInfo(_id, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &compute_capability_major, NULL);
      clGetDeviceInfo(_id, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &compute_capability_minor, NULL);
      os << "\n  CL_DEVICE_COMPUTE_CAPABILITY_NV: " << compute_capability_major << ", " << compute_capability_minor << std::endl;

      os << "  NUMBER OF MULTIPROCESSORS: " << compute_units << std::endl; // this is the same value reported by CL_DEVICE_MAX_COMPUTE_UNITS

      cl_uint regs_per_block;
      clGetDeviceInfo(_id, CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof(cl_uint), &regs_per_block, NULL);
      os << "  CL_DEVICE_REGISTERS_PER_BLOCK_NV: " << regs_per_block  << std::endl;

      cl_uint warp_size;
      clGetDeviceInfo(_id, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), &warp_size, NULL);
      os << "  CL_DEVICE_WARP_SIZE_NV: " << warp_size << std::endl;

      cl_bool gpu_overlap;
      clGetDeviceInfo(_id, CL_DEVICE_GPU_OVERLAP_NV, sizeof(cl_bool), &gpu_overlap, NULL);
      os << "  CL_DEVICE_GPU_OVERLAP_NV: " << (gpu_overlap == CL_TRUE ? "CL_TRUE" : "CL_FALSE")  << std::endl;

      cl_bool exec_timeout;
      clGetDeviceInfo(_id, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &exec_timeout, NULL);
      os << "  CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV: " << (exec_timeout == CL_TRUE ? "CL_TRUE" : "CL_FALSE")  << std::endl;

      cl_bool integrated_memory;
      clGetDeviceInfo(_id, CL_DEVICE_INTEGRATED_MEMORY_NV, sizeof(cl_bool), &integrated_memory, NULL);
      os << "  CL_DEVICE_INTEGRATED_MEMORY_NV: " << (integrated_memory == CL_TRUE ? "CL_TRUE" : "CL_FALSE")  << std::endl;
  }

  // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
  os << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>" << std::endl;
  cl_uint vec_width [6];
  clGetDeviceInfo(_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
  clGetDeviceInfo(_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
  clGetDeviceInfo(_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
  clGetDeviceInfo(_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
  clGetDeviceInfo(_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
  clGetDeviceInfo(_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
  os << "CHAR " << vec_width[0] << ", SHORT " << vec_width[1] << ", INT " << vec_width[2] << ", LONG " << vec_width[3] << ", FLOAT "
     << vec_width[4] << ", DOUBLE " << vec_width[5] << std::endl;
}

} } // namespace gpucast / namespace gl
