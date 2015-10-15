#include "./to_gray.h"
#include "../utils/cuda_helper.h"

surface<void, cudaSurfaceType2D> image;

__global__ void toGrayKernel(int image_size)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= image_size || y >= image_size)
    return;

  uchar4 color;
  surf2Dread(&color, image, x * 4, y);

  unsigned char gray = 0.2989 * color.x + 0.5870 * color.y + 0.1140 * color.z;
  color.x = gray;
  color.y = gray;
  color.z = gray;
  surf2Dwrite(color, image, x * 4, y);
}

void toGray(std::shared_ptr<CudaArrayProvider> tex, int image_size)
{
  tex->map();

  HANDLE_ERROR(
      cudaBindSurfaceToArray(image, tex->getArray(), tex->getChannelDesc()));

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(image_size, dimBlock.x), divUp(image_size, dimBlock.y), 1);

  toGrayKernel<<<dimGrid, dimBlock>>>(image_size);
  HANDLE_ERROR(cudaThreadSynchronize());

  tex->unmap();
}

