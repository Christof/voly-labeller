#if _WIN32
#pragma warning(disable : 4244)
#endif

#include "./occlusion.h"
#include "../utils/cuda_helper.h"

__global__ void occlusionKernel(cudaTextureObject_t positions,
                                cudaSurfaceObject_t output,
                                bool addToOutputValue, int width, int height,
                                int widthScale, int heightScale, int layerIndex)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float minTransparency = 0.0f;
  for (int i = 0; i < widthScale; ++i)
  {
    for (int j = 0; j < heightScale; ++j)
    {
      float4 color = tex3D<float4>(positions, x * widthScale + 0.5f + i,
                                   y * heightScale + 0.5f + j, layerIndex);
      if (color.w > minTransparency)
        minTransparency = color.w;
    }
  }

  if (addToOutputValue)
  {
    float value;
    surf2Dread(&value,  output, x * sizeof(float), y);

    float newValue = fmin(value + minTransparency, 1.0f);
    surf2Dwrite(newValue, output, x * sizeof(float), y);
  }
  else
  {
    surf2Dwrite(minTransparency, output, x * sizeof(float), y);
  }
}

namespace Placement
{

Occlusion::Occlusion(
    std::shared_ptr<CudaArrayProvider> colorProvider,
    std::shared_ptr<CudaArrayProvider> outputProvider,
    int layerIndex)
  : colorProvider(colorProvider), outputProvider(outputProvider),
  layerIndex(layerIndex)
{
}

Occlusion::~Occlusion()
{
  if (positions)
    cudaDestroyTextureObject(positions);
  if (output)
    cudaDestroySurfaceObject(output);
}

void Occlusion::addOcclusion()
{
  runKernel(true);
}

void Occlusion::calculateOcclusion()
{
  runKernel(false);
}

void Occlusion::runKernel(bool addToOutputValue)
{
  if (!positions)
    createSurfaceObjects();

  float outputWidth = outputProvider->getWidth();
  float outputHeight = outputProvider->getHeight();

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x), divUp(outputHeight, dimBlock.y),
               1);

  int widthScale = colorProvider->getWidth() / outputWidth;
  int heightScale = colorProvider->getHeight() / outputHeight;

  occlusionKernel<<<dimGrid, dimBlock>>>(positions, output, addToOutputValue,
                                         outputWidth, outputHeight,
                                         widthScale, heightScale, layerIndex);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void Occlusion::createSurfaceObjects()
{
  colorProvider->map();
  outputProvider->map();

  auto resDesc = colorProvider->getResourceDesc();
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

  colorProvider->unmap();
  outputProvider->unmap();
}

}  // namespace Placement
