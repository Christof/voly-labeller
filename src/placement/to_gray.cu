#include "./to_gray.h"
#include "../utils/cuda_helper.h"

__global__ void toGrayKernel(cudaSurfaceObject_t image, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  uchar4 color;
  surf2Dread(&color, image, x * 4, y);

  unsigned char gray = 0.2989 * color.x + 0.5870 * color.y + 0.1140 * color.z;
  color.x = gray;
  color.y = gray;
  color.z = gray;
  surf2Dwrite(color, image, x * 4, y);
}

ToGray::ToGray(std::shared_ptr<CudaArrayProvider> imageProvider)
  : imageProvider(imageProvider)
{
}

void ToGray::runKernel()
{
  imageProvider->map();
  auto resDesc = imageProvider->getResourceDesc();
  cudaCreateSurfaceObject(&image, &resDesc);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(imageProvider->getWidth(), dimBlock.x),
               divUp(imageProvider->getHeight(), dimBlock.y), 1);

  toGrayKernel<<<dimGrid, dimBlock>>>(image, imageProvider->getWidth(),
      imageProvider->getHeight());
  HANDLE_ERROR(cudaThreadSynchronize());

  imageProvider->unmap();
}

