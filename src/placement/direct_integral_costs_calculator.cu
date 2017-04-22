#include "./direct_integral_costs_calculator.h"
#include "../utils/cuda_helper.h"

__global__ void integralCosts(cudaTextureObject_t colors,
                              cudaTextureObject_t saliency,
                              float fixOcclusionPart,
                              cudaSurfaceObject_t output, int width, int height,
                              int layerIndex, int layerCount,
                              float widthScale, float heightScale)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float colorX = x * widthScale + 0.5f;
  float colorY = y * heightScale + 0.5f;

  float sum = 0.0f;

  for (int layerIndexFront = 0; layerIndexFront <= layerIndex;
       ++layerIndexFront)
    sum += tex3D<float4>(colors, colorX, colorY, layerIndexFront + 0.5f).w;

  float saliencyValue = tex2D<float>(saliency, x + 0.5f, y + 0.5f);
  float occlusionFactor = (1 - sum) *
      (1.0f - fixOcclusionPart) * saliencyValue + fixOcclusionPart;

  for (int layerIndexBehind = layerIndex + 1; layerIndexBehind < layerCount;
       ++layerIndexBehind)
  {
    sum += occlusionFactor *
           tex3D<float4>(colors, colorX, colorY, layerIndexBehind + 0.5f).w;
  }

  surf2Dwrite(sum / layerCount, output, x * sizeof(float), y);
}

__global__ void integralCostsSingleLayer(cudaTextureObject_t colors,
                                         float occlusionWeight,
                                         cudaTextureObject_t saliency,
                                         float saliencyWeight,
                                         cudaSurfaceObject_t output,
                                         int width, int height,
                                         float widthScale, float heightScale,
                                         int layerIndex)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float colorX = x * widthScale + 0.5f;
  float colorY = y * heightScale + 0.5f;
  float occlusion = tex3D<float4>(colors, colorX, colorY, 0.5f).w;

  float saliencyValue = tex2D<float>(saliency, x + 0.5f, y + 0.5f);

  float sum = occlusionWeight * occlusion + saliencyWeight * saliencyValue;

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

  float outputWidth = outputProvider->getWidth();
  float outputHeight = outputProvider->getHeight();

  float widthScale = colorProvider->getWidth() / outputWidth;
  float heightScale = colorProvider->getHeight() / outputHeight;


  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x), divUp(outputHeight, dimBlock.y),
               1);

  if (layerCount == 1)
  {
    integralCostsSingleLayer<<<dimGrid, dimBlock>>>(
        color, weights.occlusion, saliency, weights.saliency, output,
        outputWidth, outputHeight, widthScale, heightScale, layerIndex);
  }
  else
  {
    integralCosts<<<dimGrid, dimBlock>>>(
        color, saliency, weights.fixOcclusionPart, output, outputWidth,
        outputHeight, layerIndex, layerCount, widthScale, heightScale);
  }

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
