#ifndef SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#define SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

thrust::host_vector<float> algSAT(float *h_inout, int w, int h);

/**
 * \brief
 *
 *
 */
class SummedAreaTable
{
 public:
  SummedAreaTable(std::shared_ptr<CudaArrayProvider> inputImage);

  void runKernel();

 private:
  std::shared_ptr<CudaArrayProvider> inputImage;

  thrust::device_vector<float> inout;
  thrust::device_vector<float> ybar;
  thrust::device_vector<float> vhat;
  thrust::device_vector<float> ysum;
};

#endif  // SRC_PLACEMENT_SUMMED_AREA_TABLE_H_
