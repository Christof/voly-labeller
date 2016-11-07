#include "./direct_integral_costs_calculator.h"
#include "../utils/cuda_helper.h"

__global__ void integralCosts(cudaTextureObject_t colors, float occlusionWeight,
                              cudaTextureObject_t saliency,
                              float saliencyWeight, cudaSurfaceObject_t output,
                              int width, int height, int layerIndex,
                              int layerCount)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float sum = 0.0f;

  for (int layerIndexFront = 0; layerIndexFront <= layerIndex;
       ++layerIndexFront)
    sum += tex3D<float4>(colors, x + 0.5f, y + 0.5f, layerIndexFront + 0.5f).w;

  float saliencyValue = tex2D<float>(saliency, x + 0.5f, y + 0.5f);

  for (int layerIndexBehind = layerIndex + 1; layerIndexBehind < layerCount;
       ++layerIndexBehind)
  {
    sum += saliencyValue *
           tex3D<float4>(colors, x + 0.5f, y + 0.5f, layerIndexBehind + 0.5f).w;
  }

  surf2Dwrite(sum, output, x * sizeof(float), y);
}

namespace Placement
{

DirectIntegralCostsCalculator::DirectIntegralCostsCalculator(
    std::shared_ptr<CudaArrayProvider> colorProvider,
    std::shared_ptr<CudaArrayProvider> saliencyProvider,
    std::shared_ptr<CudaArrayProvider> outputProvider)
  : colorProvider(colorProvider), saliencyProvider(saliencyProvider),
    outputProvider(outputProvider)
{
}

DirectIntegralCostsCalculator::~DirectIntegralCostsCalculator()
{
  if (color)
    cudaDestroyTextureObject(color);
  if (saliency)
    cudaDestroyTextureObject(saliency);
  if (output)
    cudaDestroySurfaceObject(output);
}

void DirectIntegralCostsCalculator::runKernel(int layerIndex, int layerCount)
{
  if (!color)
    createSurfaceObjects();

  int outputWidth = outputProvider->getWidth();
  int outputHeight = outputProvider->getHeight();

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x), divUp(outputHeight, dimBlock.y),
               1);

  integralCosts<<<dimGrid, dimBlock>>>(color, weights.occlusion, saliency,
                                       weights.saliency, output, outputWidth,
                                       outputHeight, layerIndex, layerCount);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void DirectIntegralCostsCalculator::createSurfaceObjects()
{
  colorProvider->map();
  saliencyProvider->map();
  outputProvider->map();

  auto resDesc = colorProvider->getResourceDesc();
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&color, &resDesc, &texDesc, NULL);

  auto saliencyResDesc = saliencyProvider->getResourceDesc();
  cudaCreateTextureObject(&saliency, &saliencyResDesc, &texDesc, NULL);

  auto outputResDesc = outputProvider->getResourceDesc();
  cudaCreateSurfaceObject(&output, &outputResDesc);

  colorProvider->unmap();
  saliencyProvider->unmap();
  outputProvider->unmap();
}

}  // namespace Placement
