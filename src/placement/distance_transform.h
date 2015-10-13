#include <memory>
#include <thrust/device_vector.h>
#include "../utils/cuda_array_provider.h"

/**
 * \brief
 *
 *
 */
class DistanceTransform
{
 public:
  DistanceTransform(std::shared_ptr<CudaArrayProvider> inputImage,
                    std::shared_ptr<CudaArrayProvider> outputImage);

  void run();

  thrust::device_vector<float> &getResults();

 private:
  std::shared_ptr<CudaArrayProvider> inputImage;
  std::shared_ptr<CudaArrayProvider> outputImage;
  thrust::device_vector<int> computeVector;
  thrust::device_vector<float> resultVector;
  int pixelCount;
  dim3 dimBlock;
  dim3 dimGrid;
  cudaTextureObject_t inputTexture;
  cudaSurfaceObject_t outputSurface;
  void resize();
  void runInitializeKernel();
  void runStepsKernels();
  void runFinishKernel();
};

