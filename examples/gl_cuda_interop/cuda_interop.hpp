#include <cuda.h>

extern "C" void invoke_square_kernel(unsigned width, unsigned height, cudaGraphicsResource_t colorbuffer_resource);