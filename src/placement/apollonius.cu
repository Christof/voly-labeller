#include "./apollonius.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <vector>
#include "../utils/cuda_helper.h"

__global__ void seed(cudaSurfaceObject_t output, int imageSize, int labelCount,
                     float4 *seedBuffer, int *computePtr, int *idPtr,
                     int *indicesPtr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imageSize || y >= imageSize)
    return;

  int index = y * imageSize + x;

  // initialize to out of bounds
  int outIndex = (imageSize * 2) * (imageSize * 2) - 1;

  for (int i = 0; i < labelCount; i++)
  {
    float4 seedValue = seedBuffer[i];
    int4 seedValueInt =
        make_int4(static_cast<int>(seedValue.x), static_cast<int>(seedValue.y),
                  static_cast<int>(seedValue.z), static_cast<int>(seedValue.w));
    if (seedValueInt.x > 0 && x == seedValueInt.y && y == seedValueInt.z &&
        (x != 0 || y != 0))
    {
      outIndex = x + y * imageSize;
    }

    idPtr[i] = seedValueInt.x;
    indicesPtr[i] = seedValueInt.y + seedValueInt.z * imageSize;
  }

  computePtr[index] = outIndex;
}

__global__ void apolloniusStep(cudaTextureObject_t distances, int *data,
                               unsigned int step, int w, int h)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  int index = y * w + x;

  int currentNearest = data[index];
  int currentY = currentNearest / w;
  int currentX = currentNearest - currentY * w;
  float curr_w = (currentNearest < w * h)
                     ? tex2D<float>(distances, currentX + 0.5f, currentY + 0.5f)
                     : 0.0f;

  float currentDistance =
      sqrtf(static_cast<float>((x - currentX) * (x - currentX) +
                               (y - currentY) * (y - currentY))) -
      curr_w;

#pragma unroll
  for (int i = -1; i <= 1; i++)
  {
    int u = x + i * step;
    if (u < 0 || u >= w)
      continue;
#pragma unroll
    for (int j = -1; j <= 1; j += 2 - i * i)
    {
      int v = y + j * step;
      if (v < 0 || v >= h)
        continue;

      int newindex = v * w + u;
      int newNearest = data[newindex];
      int newY = newNearest / w;
      int newX = newNearest - newY * w;
      float newW = (newNearest < w * h)
                       ? tex2D<float>(distances, newX + 0.5f, newY + 0.5f)
                       : 0.0f;
      float newDistance = sqrtf(static_cast<float>((x - newX) * (x - newX) +
                                                   (y - newY) * (y - newY))) -
                          newW;

      if (newDistance < currentDistance || currentNearest >= w * h)
      {
        currentDistance = newDistance;
        currentNearest = newNearest;
      }
    }
  }

  data[index] = currentNearest;
}

/**
 * Taken from http://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
 */
__device__ float4 rgb(int maxValue, int value)
{
    float ratio = 2.0f * value / maxValue;
    float b = max(0.0f, 1 - ratio);
    float r = max(0.0f, ratio - 1);
    float g = max(0.0f, 1.0f - b - r);

    return make_float4(r, g, b, 1.0);
}

__global__ void gather(cudaSurfaceObject_t output, int imageSize,
                       int labelCount, int *nearestIndex, int *seedIds,
                       int *seedIndices)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imageSize || y >= imageSize)
    return;

  int index = y * imageSize + x;
  int labelId = -1;
  int labelIndex = nearestIndex[index];

  for (int i = 0; i < labelCount; i++)
  {
    if (labelIndex == seedIndices[i])
    {
      labelId = seedIds[i];
      break;
    }
  }

  float4 color = labelId == -1 ?
    make_float4(1, 1, 1, 1) : rgb(labelCount, labelId);

  surf2Dwrite(color, output, x * sizeof(float4), y);
}

__global__ void copyBorderIndex(int imageSize, int *source, int *destination)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int maxIndex = imageSize * imageSize - 1;

  // FIXME: corner pixels are duplicated
  if (index >= imageSize)
    return;

  // upper border from left to right
  destination[index] = source[index];

  // right border from top to bottom
  destination[imageSize + index] =
      source[(imageSize - 1) + index * imageSize];

  // bottom border from right to left
  destination[imageSize * 2 + index] = source[maxIndex - index];

  // left border from bottom to top
  destination[imageSize * 3 + index] =
      source[maxIndex - imageSize + 1 - index * imageSize];
}

namespace Placement
{

Apollonius::Apollonius(std::shared_ptr<CudaArrayProvider> distancesImage,
                       std::shared_ptr<CudaArrayProvider> outputImage,
                       std::vector<Eigen::Vector4f> labelPositions,
                       int labelCount)
  : distancesImage(distancesImage), outputImage(outputImage),
    labelCount(labelCount), seedBuffer(labelCount)
{
  imageSize = outputImage->getWidth();

  for (size_t i = 0; i < seedBuffer.size(); ++i)
  {
    auto labelPosition = labelPositions[i];
    seedBuffer[i] = make_float4(labelPosition.x(), labelPosition.y(),
                                labelPosition.z(), labelPosition.w());
    int index = static_cast<int>(labelPosition.y()) +
                static_cast<int>(labelPosition.z()) * imageSize;
    pixelIndexToLabelId[index] = labelPosition.x();
  }

  pixelCount = imageSize * imageSize;

  distancesImage->map();
  auto resDesc = distancesImage->getResourceDesc();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&distancesTexture, &resDesc, &texDesc, NULL);

  distancesImage->unmap();
}

void Apollonius::run()
{
  resize();
  outputImage->map();
  auto resDesc = outputImage->getResourceDesc();
  cudaCreateSurfaceObject(&outputSurface, &resDesc);

  dimBlock = dim3(32, 32, 1);
  dimGrid = dim3(divUp(imageSize, dimBlock.x), divUp(imageSize, dimBlock.y), 1);

  runSeedKernel();
  runStepsKernels();
  runGatherKernel();

  outputImage->unmap();
}

thrust::device_vector<int> &Apollonius::getIds()
{
  return seedIds;
}

std::vector<int> Apollonius::calculateOrdering()
{
  extractUniqueBoundaryIndices();
  updateLabelSeeds();

  size_t iterationCount = 0;
  size_t labelCount = pixelIndexToLabelId.size();
  while (extractedIndices.size() < labelCount && iterationCount < labelCount)
  {
    run();
    extractUniqueBoundaryIndices();
    updateLabelSeeds();

    ++iterationCount;
  }

  return std::vector<int>(insertionOrder.begin(), insertionOrder.end());
}

void Apollonius::resize()
{
  if (computeVector.size() != static_cast<unsigned long>(pixelCount))
  {
    computeVector.resize(pixelCount, pixelCount);
  }

  if (seedIds.size() != labelCount || seedIndices.size() != labelCount)
  {
    seedIds.resize(labelCount, -1);
    seedIndices.resize(labelCount, -1);
  }
}

void Apollonius::runSeedKernel()
{
  int *computePtr = thrust::raw_pointer_cast(computeVector.data());
  int *idPtr = thrust::raw_pointer_cast(seedIds.data());
  int *indicesPtr = thrust::raw_pointer_cast(seedIndices.data());
  float4 *seedBufferPtr = thrust::raw_pointer_cast(seedBuffer.data());

  seed<<<dimGrid, dimBlock>>>(outputSurface, imageSize, labelCount,
                              seedBufferPtr, computePtr, idPtr, indicesPtr);
  HANDLE_ERROR(cudaThreadSynchronize());
}

void Apollonius::runStepsKernels()
{
  apolloniusStep<<<dimGrid, dimBlock>>>
      (distancesTexture, thrust::raw_pointer_cast(computeVector.data()),
       1, imageSize, imageSize);

  for (int k = (imageSize / 2); k > 0; k /= 2)
  {
    apolloniusStep<<<dimGrid, dimBlock>>>
        (distancesTexture, thrust::raw_pointer_cast(computeVector.data()),
         k, imageSize, imageSize);
  }
  HANDLE_ERROR(cudaThreadSynchronize());
}

void Apollonius::runGatherKernel()
{
  int *computePtr = thrust::raw_pointer_cast(computeVector.data());
  int *seedIdsPtr = thrust::raw_pointer_cast(seedIds.data());
  int *seedIndicesPtr = thrust::raw_pointer_cast(seedIndices.data());
  gather<<<dimGrid, dimBlock>>>(outputSurface, imageSize, labelCount,
      computePtr, seedIdsPtr, seedIndicesPtr);
  HANDLE_ERROR(cudaThreadSynchronize());
}

void Apollonius::extractUniqueBoundaryIndices()
{
  const unsigned int borderSize = 4 * imageSize;

  if (orderedIndices.size() < borderSize)
  {
    orderedIndices.resize(borderSize);
  }

  dim3 dimBlock(32, 1, 1);
  dim3 dimGrid(divUp(imageSize, dimBlock.x), 1, 1);

  int *computePtr = thrust::raw_pointer_cast(computeVector.data());
  int *orderedIndicesPtr = thrust::raw_pointer_cast(orderedIndices.data());

  copyBorderIndex<<<dimGrid, dimBlock>>>(imageSize, computePtr,
      orderedIndicesPtr);

  HANDLE_ERROR(cudaThreadSynchronize());

  thrust::device_vector<int>::iterator it_found =
      thrust::unique(orderedIndices.begin(), orderedIndices.end());
  thrust::host_vector<int> uniqueIndices(orderedIndices.begin(), it_found);

  for (unsigned int i = 0; i < uniqueIndices.size(); i++)
  {
    const int uniqueIndex = uniqueIndices[i];
    if (uniqueIndex > 0)
    {
      if (extractedIndices.insert(uniqueIndex).second)
      {
        insertionOrder.push_front(pixelIndexToLabelId[uniqueIndex]);
      }
    }
  }
}

void Apollonius::updateLabelSeeds()
{
  thrust::host_vector<float4> labelsSeed = seedBuffer;
  for (size_t index = 0; index < labelsSeed.size(); ++index)
  {
    float4 seed = labelsSeed[index];
    const uint cindex =
        static_cast<int>(seed.y) + static_cast<int>(seed.z) * imageSize;

    if (extractedIndices.find(cindex) != extractedIndices.end())
    {
      if (seed.x > 0)
      {
        seed.x = -seed.x;
        labelsSeed[index] = seed;
      }
    }
  }

  seedBuffer = labelsSeed;
}

}  // namespace Placement
