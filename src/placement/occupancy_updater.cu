#include "./occupancy_updater.h"
#include <memory>
#include "../utils/cuda_helper.h"

OccupancyUpdater::OccupancyUpdater(std::shared_ptr<CudaArrayProvider> occupancy)
  : occupancy(occupancy)
{
}

OccupancyUpdater::~OccupancyUpdater()
{
  if (surface)
    cudaDestroySurfaceObject(surface);
}

__global__ void addLabelOccupancy(cudaSurfaceObject_t surface, int left,
                                  int top, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  surf2Dwrite(1.0f, surface, (left + x) * sizeof(float), top + y);
}

void OccupancyUpdater::addLabel(int x, int y, int width, int height)
{
  if (!surface)
    createSurface();

  int outputWidth = occupancy->getWidth();
  int outputHeight = occupancy->getHeight();

  int left = x - width / 2;
  if (left < 0)
  {
    width += left;
    left = 0;
  }
  if (left + width >= outputWidth)
  {
    width = outputWidth - left;
  }

  int top = y - height / 2;
  if (top < 0)
  {
    height += top;
    top = 0;
  }
  if (top + height >= outputHeight)
  {
    height = outputHeight - top;
  }

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(width, dimBlock.x),
               divUp(height, dimBlock.y), 1);

  addLabelOccupancy<<<dimGrid, dimBlock>>>(surface, left, top, width,
      height);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void OccupancyUpdater::createSurface()
{
  occupancy->map();

  auto resourceDesc = occupancy->getResourceDesc();
  cudaCreateSurfaceObject(&surface, &resourceDesc);

  occupancy->unmap();
}

