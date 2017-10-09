#include <iostream>
#include <algorithm>
#include <array>

#include <cuda.h>
#include <cuda_helper.hpp>
#include <device_types.h>
#include <vector_types.h>


surface<void, cudaSurfaceType2D> out_color_image;



///////////////////////////////////////////////////////////////////////////////
template <typename function_type>
inline std::size_t get_kernel_workitems(function_type kernel)
{
  cudaFuncAttributes  kernel_attribs;
  cudaError_t err = cudaFuncGetAttributes(&kernel_attribs, kernel);

  if (err != cudaSuccess)
  {
    std::cerr << "Cannot retrieve kernel attributes" << std::endl;
  }

  float max_workitems = std::sqrt(float(kernel_attribs.maxThreadsPerBlock));
  float max_workitems_exp2 = std::log(max_workitems) / std::log(2.0f);

  std::size_t workitems = std::size_t(std::pow(2.0f, std::floor(max_workitems_exp2)));
  workitems = std::min(std::size_t(256 * 256), workitems);
  workitems = std::max(workitems, std::size_t(1));
  std::cout << "Work items for kernel : " << workitems << std::endl;

  return std::max(workitems, std::size_t(1));
}


///////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void square_kernel(int width, int height)
{
  int sx = blockIdx.x*blockDim.x + threadIdx.x;
  int sy = blockIdx.y*blockDim.y + threadIdx.y;

  if (sx >= width || sy >= height)
  {
    return;
  }

  int2  coords = make_int2(sx, sy);

  float4 color = make_float4(1.0, 1.0, 1.0, 1.0);
  if ((sx / 10 + sy / 10) % 2) {
    color = make_float4(0.0, 0.0, 0.0, 1.0);
  }

  surf2Dwrite(color, out_color_image, coords.x*sizeof(float4), coords.y);
}


///////////////////////////////////////////////////////////////////////////////
extern "C" void invoke_square_kernel (unsigned width, unsigned height, cudaGraphicsResource_t colorbuffer_resource )
{
  cudaGraphicsResource_t cuda_resources[] = { colorbuffer_resource };

  // map gl resource and retrieve device_ptr
  checkCudaErrors(cudaGraphicsMapResources(1, cuda_resources, 0));

  cudaArray_t resource_array;
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&resource_array, colorbuffer_resource, 0, 0));

  cudaChannelFormatDesc desc;
  checkCudaErrors(cudaGetChannelDesc(&desc, resource_array));

  checkCudaErrors(cudaBindSurfaceToArray(&out_color_image, resource_array, &desc));

  { // raycast kernel
    int min_grid_size;
    int block_size;
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, square_kernel, 0, 0));
    
    const int block_width = 16;

    dim3 block(block_width, block_width, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    
    square_kernel << <grid, block >> >(width, height);
  }

  // map gl resource and release device_ptr
  checkCudaErrors(cudaGraphicsUnmapResources(1, cuda_resources, 0));
}

