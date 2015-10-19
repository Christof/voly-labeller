#include "./apollonius.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
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
  float4 outValue = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

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
      outValue =
          make_float4(seedValue.x / (labelCount + 1),
                      seedValueInt.y / static_cast<float>(imageSize),
                      seedValueInt.z / static_cast<float>(imageSize), 1.0f);

      outIndex = x + y * imageSize;
    }
    idPtr[i] = seedValueInt.x;
    indicesPtr[i] = seedValueInt.y + seedValueInt.z * imageSize;
  }

  computePtr[index] = outIndex;
  surf2Dwrite(outValue, output, x * sizeof(float4), y);
}

__global__ void apolloniusStep(int *data, float *occupancy, unsigned int step,
                               int w, int h)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  int index = y * w + x;

  int currentNearest = data[index];
  int currentY = currentNearest / w;
  int currentX = currentNearest - currentY * w;
  float curr_w = (currentNearest < w * h) ? occupancy[currentNearest] : 0.0f;

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
      float newW = (newNearest < w * h) ? occupancy[newNearest] : 0.0f;
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

__global__ void gather(cudaSurfaceObject_t output, int imageSize,
                       int labelCount, int *nearestIndex, int *seedIds,
                       int *seedIndices)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imageSize || y >= imageSize)
    return;

  int index = y * imageSize + x;
  float4 color;
  int labelId = 100;
  int labelIndex = nearestIndex[index];

  for (int i = 0; i < labelCount; i++)
  {
    if (labelIndex == seedIndices[i])
    {
      labelId = seedIds[i];
      break;
    }
  }

  switch (labelId)
  {
  case 0:
    color = make_float4(0.0, 0.0, 0.0, 1.0);
    break;
  case 1:
    color = make_float4(1.0, 0.0, 0.0, 1.0);
    break;
  case 2:
    color = make_float4(0.0, 1.0, 0.0, 1.0);
    break;
  case 3:
    color = make_float4(0.0, 0.0, 1.0, 1.0);
    break;
  case 4:
    color = make_float4(1.0, 1.0, 0.0, 1.0);
    break;
  case 5:
    color = make_float4(0.0, 1.0, 1.0, 1.0);
    break;
  case 6:
    color = make_float4(1.0, 0.0, 1.0, 1.0);
    break;
  case 7:
    color = make_float4(1.0, 1.0, 1.0, 1.0);
    break;
  default:
    color = make_float4(0.5, 0.5, 0.5, 1.0);
  }
  surf2Dwrite(color, output, x * sizeof(float4), y);
}

Apollonius::Apollonius(std::shared_ptr<CudaArrayProvider> inputImage,
                       thrust::device_vector<float4> &seedBuffer,
                       thrust::device_vector<float> &distances, int labelCount)
  : inputImage(inputImage), seedBuffer(seedBuffer), distances(distances),
    labelCount(labelCount)
{
  imageSize = inputImage->getWidth();
  pixelCount = imageSize * imageSize;
}

void Apollonius::run()
{
  resize();
  inputImage->map();
  auto resDesc = inputImage->getResourceDesc();
  cudaCreateSurfaceObject(&outputSurface, &resDesc);

  dimBlock = dim3(32, 32, 1);
  dimGrid = dim3(divUp(imageSize, dimBlock.x), divUp(imageSize, dimBlock.y), 1);

  runSeedKernel();
  runStepsKernels();
  runGatherKernel();

  inputImage->unmap();
}

thrust::device_vector<int> &Apollonius::getIds()
{
  return seedIds;
}

void Apollonius::resize()
{
  if (computeVector.size() != static_cast<unsigned long>(pixelCount))
  {
    computeVector.resize(pixelCount, pixelCount);
  }

  if (seedIds.size() != MAX_LABELS || seedIndices.size() != MAX_LABELS)
  {
    seedIds.resize(MAX_LABELS, -1);
    seedIndices.resize(MAX_LABELS, -1);
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
      (thrust::raw_pointer_cast(computeVector.data()),
       thrust::raw_pointer_cast(distances.data()), 1, imageSize, imageSize);

  for (int k = (imageSize / 2); k > 0; k /= 2)
  {
    apolloniusStep<<<dimGrid, dimBlock>>>
        (thrust::raw_pointer_cast(computeVector.data()),
         thrust::raw_pointer_cast(distances.data()), k, imageSize, imageSize);
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

thrust::device_vector<float4>
Apollonius::createSeedBufferFromLabels(std::vector<Label> labels,
                                       Eigen::Matrix4f viewProjection,
                                       Eigen::Vector2i size)
{
  thrust::host_vector<float4> result;
  for (auto &label : labels)
  {
    Eigen::Vector4f pos =
        viewProjection * Eigen::Vector4f(label.anchorPosition.x(),
                                         label.anchorPosition.y(),
                                         label.anchorPosition.z(), 1);
    float x = (pos.x() / pos.w() * 0.5f + 0.5f) * size.x();
    float y = (pos.y() / pos.w() * 0.5f + 0.5f) * size.y();
    result.push_back(make_float4(label.id, x, y, 1));
  }

  return result;
}

