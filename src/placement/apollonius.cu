#include "./apollonius.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "../utils/cuda_helper.h"

surface<void, cudaSurfaceType2D> surfaceWrite;
texture<float, 2, cudaReadModeElementType> depthTexture;

__global__ void seed(int imageSize, int labelCount, float4 *seedbuffer,
                     int *thrustptr, int *idptr, int *idxptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imageSize || y >= imageSize)
    return;

  int index = y * imageSize + x;
  float4 outval = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  float4 seedval = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  // initialize to out of bounds
  int outindex = (imageSize * 2) * (imageSize * 2) - 1;

  for (int i = 0; i < labelCount; i++)
  {
    float4 seedval = seedbuffer[i];
    if (int(seedval.x) > 0 && x == int(seedval.y) && y == int(seedval.z) &&
        (x != 0 || y != 0))
    {
      outval = make_float4(seedval.x / (labelCount + 1),
                           int(seedval.y) / float(imageSize),
                           int(seedval.z) / float(imageSize), 1.0f);

      // index for thrust computation =
      outindex = x + y * imageSize;
    }
    idptr[i] = int(seedval.x);
    idxptr[i] = int(seedval.y) + int(seedval.z) * imageSize;
  }

  thrustptr[index] = outindex;
  surf2Dwrite<float4>(outval, surfaceWrite, x * sizeof(float4), y);
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

  float currentDistance = sqrtf(float((x - currentX) * (x - currentX) +
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
      float newDistance =
          sqrtf(float((x - newX) * (x - newX) + (y - newY) * (y - newY))) -
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

__global__ void gather(int imageSize, int labelCount, int *thrustptr,
                       int *seedidptr, int *seedidxptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imageSize || y >= imageSize)
    return;
  int index = y * imageSize + x;
  float4 color;
  int labelID = 100;
  int labelIndex = thrustptr[index];

  for (int i = 0; i < labelCount; i++)
  {
    if (labelIndex == seedidxptr[i])
    {
      labelID = seedidptr[i];
      break;
    }
  }

  switch (labelID)
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
  surf2Dwrite<float4>(color, surfaceWrite, x * sizeof(float4), y);
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
  dimBlock = dim3(32, 32, 1);
  dimGrid = dim3(divUp(imageSize, dimBlock.x), divUp(imageSize, dimBlock.y), 1);

  runSeedKernel();
  runStepsKernels();
  runGatherKernel();

  inputImage->unmap();
}

void Apollonius::resize()
{
  if (computeVector.size() != static_cast<unsigned long>(pixelCount))
  {
    computeVector.resize(pixelCount, pixelCount);
    computeVectorTemp.resize(pixelCount, pixelCount);
  }

  if (seedIds.size() != MAX_LABELS || seedIndices.size() != MAX_LABELS)
  {
    seedIds.resize(MAX_LABELS, -1);
    seedIndices.resize(MAX_LABELS, -1);
  }
}

void Apollonius::runSeedKernel()
{
  int *raw_ptr = thrust::raw_pointer_cast(computeVector.data());
  int *idptr = thrust::raw_pointer_cast(seedIds.data());
  int *idxptr = thrust::raw_pointer_cast(seedIndices.data());
  float4 *seedBufferPtr = thrust::raw_pointer_cast(seedBuffer.data());

  HANDLE_ERROR(cudaBindSurfaceToArray(surfaceWrite, inputImage->getArray(),
                                      inputImage->getChannelDesc()));

  seed<<<dimGrid, dimBlock>>>(imageSize, labelCount, seedBufferPtr, raw_ptr, 
      idptr, idxptr);
  HANDLE_ERROR(cudaThreadSynchronize());
}

void Apollonius::runStepsKernels()
{
  computeVectorTemp = computeVector;
  apolloniusStep<<<dimGrid, dimBlock>>>
      (thrust::raw_pointer_cast(computeVector.data()),
       thrust::raw_pointer_cast(distances.data()), 1, imageSize,
       imageSize);

  for (int k = (imageSize / 2); k > 0; k /= 2)
  {
    apolloniusStep<<<dimGrid, dimBlock>>>(
        thrust::raw_pointer_cast(computeVector.data()),
        thrust::raw_pointer_cast(distances.data()), k, imageSize,
        imageSize);
  }
  HANDLE_ERROR(cudaThreadSynchronize());
}

void Apollonius::runGatherKernel()
{
  int *raw_ptr = thrust::raw_pointer_cast(computeVector.data());
  int *idptr = thrust::raw_pointer_cast(seedIds.data());
  int *idxptr = thrust::raw_pointer_cast(seedIndices.data());
  gather<<<dimGrid, dimBlock>>>(imageSize, labelCount, raw_ptr,
      idptr, idxptr);
  HANDLE_ERROR(cudaThreadSynchronize());
}

