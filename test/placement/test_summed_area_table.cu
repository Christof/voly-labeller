#include <Eigen/Core>
#include "../../src/utils/cuda_helper.h"
#include "../cuda_array_mapper.h"
#include "../../src/placement/summed_area_table.h"
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>

int sumUsingThrustReduce()
{
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;

  std::cout << "Thrust v" << major << "." << minor << std::endl;
  thrust::host_vector<int> values;
  values.push_back(1);
  values.push_back(2);
  values.push_back(3);

  thrust::device_vector<int> deviceValues = values;

  return thrust::reduce(deviceValues.begin(), deviceValues.end(), 0,
                        thrust::plus<int>());
}

__global__ void sumCuda(const int *values, int size, int *result)
{
  *result = 0;
  for (int i = 0; i < size; ++i)
    *result += values[i];
}

int sumUsingCuda()
{
  thrust::host_vector<int> values;
  values.push_back(1);
  values.push_back(2);
  values.push_back(3);

  thrust::device_vector<int> deviceValues = values;

  thrust::device_vector<int> deviceResult(1);

  int *valuesPtr = thrust::raw_pointer_cast(&deviceValues[0]);
  int *resultPtr = thrust::raw_pointer_cast(deviceResult.data());
  sumCuda<<<1, 1>>>(valuesPtr, deviceValues.size(), resultPtr);

  thrust::host_vector<int> result = deviceResult;

  return result[0];
}

unsigned int toGrayUsingCuda(unsigned int value)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  std::vector<unsigned int> data = { value };
  auto arrayProvider =
      std::make_shared<CudaArrayMapper<unsigned int>>(1, 1, data, channelDesc);

  toGray(arrayProvider, 1);

  auto resultVector = arrayProvider->copyDataFromGpu();

  return resultVector[0];
}

