#include <memory>
#include <thrust/device_vector.h>
#include "../utils/cuda_array_provider.h"

#define MAX_LABELS 256


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

  thrust::device_vector<float>& getResults();
 private:
  std::shared_ptr<CudaArrayProvider> inputImage;
  std::shared_ptr<CudaArrayProvider> outputImage;
  thrust::device_vector<int> computeVector;
  thrust::device_vector<float> resultVector;
  int pixelCount;
  dim3 dimBlock;
  dim3 dimGrid;
  // cudaTextureObject_t inputTexture;
  void resize();
  void runInitializeKernel();
  void runStepsKernels();
  void runFinishKernel();
};

void
cudaJFADistanceTransformThrust(std::shared_ptr<CudaArrayProvider> inputImage,
                               std::shared_ptr<CudaArrayProvider> outputImage,
                               int image_size, int screen_size_x,
                               int screen_size_y,
                               thrust::device_vector<int> &compute_vector,
                               thrust::device_vector<float> &result_vector);

void cudaJFADistanceTransformThrust(
    cudaArray_t inputImageArray, cudaChannelFormatDesc inputImageDesc,
    cudaArray_t outputImageArray, int image_size, int screen_size_x,
    int screen_size_y, thrust::device_vector<int> &compute_vector,
    thrust::device_vector<float> &result_vector);
