#include "./occupancy.h"
#include "../utils/cuda_helper.h"

__global__ void occupancyKernel(cudaTextureObject_t positions,
    cudaSurfaceObject_t output, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float4 position = tex2D<float4>(positions, x + 0.5f, y + 0.5f);

  float outputValue = position.z < 1.0f ? 1.0f : 0.0f;
  surf2Dwrite(outputValue, output, x * sizeof(float), y);
}

Occupancy::Occupancy(std::shared_ptr<CudaArrayProvider> positionProvider,
                     std::shared_ptr<CudaArrayProvider> outputProvider)
  : positionProvider(positionProvider), outputProvider(outputProvider)
{
}

void Occupancy::runKernel()
{
  positionProvider->map();
  auto resDesc = positionProvider->getResourceDesc();
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&positions, &resDesc, &texDesc, NULL);

  outputProvider->map();
  auto outputResDesc = outputProvider->getResourceDesc();
  cudaCreateSurfaceObject(&output, &outputResDesc);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(positionProvider->getWidth(), dimBlock.x),
               divUp(positionProvider->getHeight(), dimBlock.y), 1);

  occupancyKernel<<<dimGrid, dimBlock>>>(positions, output,
      positionProvider->getWidth(), positionProvider->getHeight());
  HANDLE_ERROR(cudaThreadSynchronize());

  cudaDestroyTextureObject(positions);
  cudaDestroySurfaceObject(output);

  positionProvider->unmap();
  outputProvider->unmap();
}
