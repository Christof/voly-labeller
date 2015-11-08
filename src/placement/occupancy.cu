#include "./occupancy.h"
#include "../utils/cuda_helper.h"

__global__ void occupancyKernel(cudaTextureObject_t positions,
                                cudaSurfaceObject_t output, int width,
                                int height, float widthScale, float heightScale)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float4 position =
      tex2D<float4>(positions, x * widthScale + 0.5f, y * heightScale + 0.5f);

  float outputValue = position.z;
  surf2Dwrite(outputValue, output, x * sizeof(float), y);
}

Occupancy::Occupancy(std::shared_ptr<CudaArrayProvider> positionProvider,
                     std::shared_ptr<CudaArrayProvider> outputProvider)
  : positionProvider(positionProvider), outputProvider(outputProvider)
{
}

Occupancy::~Occupancy()
{
  if (positions)
    cudaDestroyTextureObject(positions);
  if (output)
    cudaDestroySurfaceObject(output);
}

void Occupancy::runKernel()
{
  if (!positions)
    createSurfaceObjects();

  float outputWidth = outputProvider->getWidth();
  float outputHeight = outputProvider->getHeight();

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x),
               divUp(outputHeight, dimBlock.y), 1);

  float widthScale = positionProvider->getWidth() / outputWidth;
  float heightScale = positionProvider->getHeight() / outputHeight;

  occupancyKernel<<<dimGrid, dimBlock>>>(positions, output,
      outputWidth, outputHeight, widthScale, heightScale);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void Occupancy::createSurfaceObjects()
{
  positionProvider->map();
  outputProvider->map();

  auto resDesc = positionProvider->getResourceDesc();
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&positions, &resDesc, &texDesc, NULL);

  auto outputResDesc = outputProvider->getResourceDesc();
  cudaCreateSurfaceObject(&output, &outputResDesc);

  positionProvider->unmap();
  outputProvider->unmap();
}

