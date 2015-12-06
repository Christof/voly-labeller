#ifndef SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#define SRC_PLACEMENT_SUMMED_AREA_TABLE_H_

#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

/**
 * \brief Calculates the summed area table of a given float image
 *
 * Implementation taken from
 * Nehab, D., Maximo, A., Lima, R. S., & Hoppe, H. (2011).
 * GPU-efficient recursive filtering and summed-area tables.
 * ACM Transactions on Graphics, 30(6), 1.
 * http://doi.org/10.1145/2070781.2024210.
 *
 * Code adapted from https://github.com/andmax/gpufilter
 */
class SummedAreaTable
{
 public:
  explicit SummedAreaTable(std::shared_ptr<CudaArrayProvider> inputImage);

  void runKernel();

  thrust::device_vector<float> &getResults();

 private:
  std::shared_ptr<CudaArrayProvider> inputImage;

  thrust::device_vector<float> inout;
  thrust::device_vector<float> ybar;
  thrust::device_vector<float> vhat;
  thrust::device_vector<float> ysum;
};

#endif  // SRC_PLACEMENT_SUMMED_AREA_TABLE_H_
