#include "./integral_costs_calculator.h"
#include "../utils/cuda_helper.h"

__global__ void sumWeightedCosts(cudaTextureObject_t occlusion,
                                 float occlusionWeight,
                                 cudaTextureObject_t saliency,
                                 float saliencyWeight,
                                 cudaSurfaceObject_t output, int width,
                                 int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float occlusionValue = tex2D<float>(occlusion, x + 0.5f, y + 0.5f);
  float saliencyValue = tex2D<float>(saliency, x + 0.5f, y + 0.5f);

  float sum = occlusionWeight * occlusionValue +
    saliencyWeight * saliencyValue;

  surf2Dwrite(sum, output, x * sizeof(float), y);
}

namespace Placement
{

IntegralCostsCalculator::IntegralCostsCalculator(
    std::shared_ptr<CudaArrayProvider> occlusionProvider,
    std::shared_ptr<CudaArrayProvider> saliencyProvider,
    std::shared_ptr<CudaArrayProvider> outputProvider)
  : occlusionProvider(occlusionProvider), saliencyProvider(saliencyProvider),
    outputProvider(outputProvider)
{
}

IntegralCostsCalculator::~IntegralCostsCalculator()
{
  if (occlusion)
    cudaDestroyTextureObject(occlusion);
  if (saliency)
    cudaDestroyTextureObject(saliency);
  if (output)
    cudaDestroySurfaceObject(output);
}

void IntegralCostsCalculator::runKernel()
{
  if (!occlusion)
    createSurfaceObjects();

  int outputWidth = outputProvider->getWidth();
  int outputHeight = outputProvider->getHeight();

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x), divUp(outputHeight, dimBlock.y),
               1);

  float occlusionWeight = 1.0f;
  float saliencyWeight = 1.0f;

  sumWeightedCosts<<<dimGrid, dimBlock>>>(occlusion, occlusionWeight,
                                          saliency, saliencyWeight, output,
                                          outputWidth, outputHeight);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void IntegralCostsCalculator::createSurfaceObjects()
{
  occlusionProvider->map();
  saliencyProvider->map();
  outputProvider->map();

  auto resDesc = occlusionProvider->getResourceDesc();
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&occlusion, &resDesc, &texDesc, NULL);

  auto saliencyResDesc = saliencyProvider->getResourceDesc();
  cudaCreateTextureObject(&saliency, &saliencyResDesc, &texDesc, NULL);

  auto outputResDesc = outputProvider->getResourceDesc();
  cudaCreateSurfaceObject(&output, &outputResDesc);

  occlusionProvider->unmap();
  saliencyProvider->unmap();
  outputProvider->unmap();
}

}  // namespace Placement
