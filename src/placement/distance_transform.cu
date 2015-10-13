#include "./distance_transform.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "../utils/cuda_helper.h"

/**
 * \brief Initializes the distance transform
 *
 * The value from the inputTexture is read. If it is larger than or equel to
 * 0.99 the data value is set to the index. Otherwise it is set to the given
 * outlier value.
 */
__global__ void initializeForDistanceTransform(cudaTextureObject_t input,
                                               int width, int height,
                                               float xscale, float yscale,
                                               int outlierValue, int *data)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int index = y * width + x;

  float pixelValue = tex2D<float>(input, x * xscale + 0.5f, y * yscale + 0.5f);

  data[index] = pixelValue >= 0.99f ? index : outlierValue;
}

__global__ void distanceTransformStep(int *data, unsigned int step, int width,
                                      int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int index = y * width + x;

  int currentNearest = data[index];
  int currentY = currentNearest / width;
  int currentX = currentNearest - currentY * width;
  int currentDistance =
      (x - currentX) * (x - currentX) + (y - currentY) * (y - currentY);

  #pragma unroll
  for (int i = -1; i <= 1; i++)
  {
    int u = x + i * step;
    if (u < 0 || u >= width)
      continue;

    #pragma unroll
    for (int j = -1; j <= 1; j += 2 - i * i)
    {
      int v = y + j * step;
      if (v < 0 || v >= height)
        continue;

      int newIndex = v * width + u;
      int newNearest = data[newIndex];
      int newY = newNearest / width;
      int newX = newNearest - newY * width;
      int newDistance = (x - newX) * (x - newX) + (y - newY) * (y - newY);

      if (newDistance < currentDistance)
      {
        currentDistance = newDistance;
        currentNearest = newNearest;
      }
    }
  }

  data[index] = currentNearest;
}

__global__ void distanceTransformFinish(cudaSurfaceObject_t output, int width,
                                        int height, int *data, float *result)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int index = y * width + x;
  int voronoival = data[index];

  int ty = voronoival / height;
  int tx = voronoival - ty * width;

  float sqdist = ((tx - x) * (tx - x) + (ty - y) * (ty - y));
  float distf = sqrtf(sqdist);

  result[index] = distf;

  // write to texture for debugging
  float4 color =
      make_float4(16.0f * distf / width, 16.0f * distf / width,
                  16.0f * distf / width, 1.0f);
  surf2Dwrite(color, output, x * sizeof(float4), y);
}

DistanceTransform::DistanceTransform(
    std::shared_ptr<CudaArrayProvider> inputImage,
    std::shared_ptr<CudaArrayProvider> outputImage)
  : inputImage(inputImage), outputImage(outputImage)

{
}

void DistanceTransform::resize()
{
  pixelCount = outputImage->getWidth() * outputImage->getHeight();
  if (computeVector.size() != static_cast<unsigned long>(pixelCount))
  {
    computeVector.resize(pixelCount, pixelCount);
    resultVector.resize(pixelCount, pixelCount);
  }
}

void DistanceTransform::run()
{
  resize();
  inputImage->map();

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = inputImage->getArray();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&inputTexture, &resDesc, &texDesc, NULL);

  outputImage->map();
  struct cudaResourceDesc outputResDesc;
  memset(&outputResDesc, 0, sizeof(outputResDesc));
  outputResDesc.resType = cudaResourceTypeArray;
  outputResDesc.res.array.array = outputImage->getArray();

  cudaCreateSurfaceObject(&outputSurface, &outputResDesc);

  dimBlock = dim3(32, 32, 1);
  dimGrid = dim3(divUp(inputImage->getWidth(), dimBlock.x),
               divUp(inputImage->getHeight(), dimBlock.y), 1);

  runInitializeKernel();
  runStepsKernels();
  runFinishKernel();

  cudaDestroyTextureObject(inputTexture);
  cudaDestroySurfaceObject(outputSurface);
  inputImage->unmap();
}

thrust::device_vector<float> &DistanceTransform::getResults()
{
  return resultVector;
}

void DistanceTransform::runInitializeKernel()
{
  int *computePtr = thrust::raw_pointer_cast(computeVector.data());

  // read depth buffer and initialize distance transform computation
  float xScale =
      static_cast<float>(inputImage->getWidth()) / outputImage->getWidth();
  float yScale =
      static_cast<float>(inputImage->getHeight()) / outputImage->getHeight();
  int outlierValue =
      (outputImage->getWidth() * 2) * (outputImage->getHeight() * 2) - 1;

  initializeForDistanceTransform<<<dimGrid, dimBlock>>>(inputTexture,
      outputImage->getWidth(), outputImage->getHeight(), xScale, yScale,
      outlierValue, computePtr);
  HANDLE_ERROR(cudaThreadSynchronize());
}

void DistanceTransform::runStepsKernels()
{
  int *computePtr = thrust::raw_pointer_cast(computeVector.data());
  distanceTransformStep<<<dimGrid, dimBlock>>>(
      computePtr, 1, outputImage->getWidth(), outputImage->getHeight());

  for (int k = (outputImage->getWidth() / 2); k > 0; k /= 2)
  {
    distanceTransformStep<<<dimGrid, dimBlock>>>(
        computePtr, k, outputImage->getWidth(), outputImage->getHeight());
  }

  HANDLE_ERROR(cudaThreadSynchronize());
}

void DistanceTransform::runFinishKernel()
{
  // kernel which maps color to distance transform result
  int *computePtr = thrust::raw_pointer_cast(computeVector.data());
  float *resultPtr = thrust::raw_pointer_cast(resultVector.data());

  distanceTransformFinish<<<dimGrid, dimBlock>>>(outputSurface,
      outputImage->getWidth(), outputImage->getHeight(), computePtr, resultPtr);

  HANDLE_ERROR(cudaThreadSynchronize());
}

