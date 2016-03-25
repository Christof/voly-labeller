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

  float4 leftUpper = tex2D<float4>(input, x - 1 + 0.5f, y - 1 + 0.5f);
  float4 left = tex2D<float4>(input, x - 1 + 0.5f, y + 0.5f);
  float4 leftLower = tex2D<float4>(input, x - 1 + 0.5f, y + 1 + 0.5f);

  float4 upper = tex2D<float4>(input, x + 0.5f, y - 1 + 0.5f);
  float4 lower = tex2D<float4>(input, x + 0.5f, y + 1 + 0.5f);

  float4 rightUpper = tex2D<float4>(input, x + 1 + 0.5f, y - 1 + 0.5f);
  float4 right = tex2D<float4>(input, x + 1 + 0.5f, y + 0.5f);
  float4 rightLower = tex2D<float4>(input, x + 1 + 0.5f, y + 1 + 0.5f);

  float resultX = -leftUpper.x - 2.0f * left.x - leftLower.x +
                  rightUpper.x + 2.0f * right.x + rightLower.x;
  float resultY = -leftUpper.x - 2.0f * upper.x - rightUpper.x +
                 leftLower.x + 2.0f * lower.x + rightLower.x;

  float magnitudeSquared = resultX * resultX + resultY * resultY;
  surf2Dwrite(magnitudeSquared, output, x * sizeof(float), y);
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
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
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
