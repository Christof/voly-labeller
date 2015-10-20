#include <thrust/device_vector.h>

typedef float (*CalculateCostElement)();

__device__ float calculateCost1()
{
  return 1.0f;
}

__device__ float calculateCost2()
{
  return 2.0f;
}

__device__ float calculateCost3()
{
  return 3.0f;
}

__device__ CalculateCostElement cost1 = calculateCost1;
__device__ CalculateCostElement cost2 = calculateCost2;
__device__ CalculateCostElement cost3 = calculateCost3;

__global__ void calculateCosts(CalculateCostElement *costElements, int count,
    float* costs)
{
  float cost;
  for (int i = 0; i < count; ++i)
    cost += costElements[i]();

  costs[0] = cost;
}

float calculateCosts()
{
  int count = 2;
  CalculateCostElement hostFunctions[2];
  cudaMemcpyFromSymbol(&hostFunctions[0], cost1, sizeof(CalculateCostElement), 0,
                       cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostFunctions[1], cost2, sizeof(CalculateCostElement), 0,
                       cudaMemcpyDeviceToHost);

  CalculateCostElement *deviceFunctions;
  cudaMalloc(&deviceFunctions, count * sizeof(CalculateCostElement));
  cudaMemcpy(deviceFunctions, hostFunctions, count * sizeof(CalculateCostElement),
             cudaMemcpyHostToDevice);

  thrust::device_vector<float> costs(1);

  calculateCosts<<<1, 1>>>
      (deviceFunctions, count, thrust::raw_pointer_cast(costs.data()));

  thrust::host_vector<float> results = costs;

  return results[0];
}

