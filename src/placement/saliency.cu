#include "./saliency.h"
#include "../utils/cuda_helper.h"

__global__ void saliencyKernel(cudaTextureObject_t input,
                               cudaSurfaceObject_t output, int width,
                               int height, int widthScale, int heightScale)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float maxDepth = -1.0f;
  for (int i = 0; i < widthScale; ++i)
  {
    for (int j = 0; j < heightScale; ++j)
    {
      float4 position = tex2D<float4>(input, x * widthScale + 0.5f + i,
                                      y * heightScale + 0.5f + j);
      if (position.z > maxDepth)
        maxDepth = position.z;
    }
  }

  float outputValue = 1.0f - maxDepth;
  surf2Dwrite(outputValue, output, x * sizeof(float), y);
}

namespace Placement
{

Saliency::Saliency(std::shared_ptr<CudaArrayProvider> inputProvider,
                   std::shared_ptr<CudaArrayProvider> outputProvider)
  : inputProvider(inputProvider), outputProvider(outputProvider)
{
}

Saliency::~Saliency()
{
  if (input)
    cudaDestroyTextureObject(input);
  if (output)
    cudaDestroySurfaceObject(output);
}

void Saliency::runKernel()
{
  if (!input)
    createSurfaceObjects();

  float outputWidth = outputProvider->getWidth();
  float outputHeight = outputProvider->getHeight();

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x), divUp(outputHeight, dimBlock.y),
               1);

  int widthScale = inputProvider->getWidth() / outputWidth;
  int heightScale = inputProvider->getHeight() / outputHeight;

  saliencyKernel<<<dimGrid, dimBlock>>>
      (input, output, outputWidth, outputHeight, widthScale, heightScale);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void Saliency::createSurfaceObjects()
{
  inputProvider->map();
  outputProvider->map();

  auto resDesc = inputProvider->getResourceDesc();
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&input, &resDesc, &texDesc, NULL);

  auto outputResDesc = outputProvider->getResourceDesc();
  cudaCreateSurfaceObject(&output, &outputResDesc);

  inputProvider->unmap();
  outputProvider->unmap();
}

}  // namespace Placement
