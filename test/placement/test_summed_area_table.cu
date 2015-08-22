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

