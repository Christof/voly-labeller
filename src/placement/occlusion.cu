#include "./occlusion.h"
#include "../utils/cuda_helper.h"

__global__ void occupancyKernel(cudaTextureObject_t positions,
                                cudaSurfaceObject_t output, int width,
                                int height, int widthScale, int heightScale)
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
      float4 color = tex2D<float4>(positions, x * widthScale + 0.5f + i,
                                      y * heightScale + 0.5f + j);
      if (color.w > minTransparency)
        minTransparency = color.w;
    }
  }

  surf2Dwrite(minTransparency, output, x * sizeof(float), y);
}

namespace Placement
{

Occlusion::Occlusion(std::vector<std::shared_ptr<CudaArrayProvider>> colorProviders,
                     std::shared_ptr<CudaArrayProvider> outputProvider)
  : colorProviders(colorProviders), outputProvider(outputProvider)
{
}

Occlusion::~Occlusion()
{
  if (positions)
    cudaDestroyTextureObject(positions);
  if (output)
    cudaDestroySurfaceObject(output);
}

void Occlusion::runKernel()
{
  if (!positions)
    createSurfaceObjects();

  float outputWidth = outputProvider->getWidth();
  float outputHeight = outputProvider->getHeight();

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x), divUp(outputHeight, dimBlock.y),
               1);

  int widthScale = colorProviders[0]->getWidth() / outputWidth;
  int heightScale = colorProviders[0]->getHeight() / outputHeight;

  occupancyKernel<<<dimGrid, dimBlock>>>(positions, output,
      outputWidth, outputHeight, widthScale, heightScale);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void Occlusion::createSurfaceObjects()
{
  colorProviders[0]->map();
  outputProvider->map();

  auto resDesc = colorProviders[0]->getResourceDesc();
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

  colorProviders[0]->unmap();
  outputProvider->unmap();
}

}  // namespace Placement
