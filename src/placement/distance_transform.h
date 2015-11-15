#ifndef SRC_PLACEMENT_DISTANCE_TRANSFORM_H_

#define SRC_PLACEMENT_DISTANCE_TRANSFORM_H_

#include <thrust/device_vector.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

/**
 * \brief Calculates for each pixel distance to nearest pixel with 
 * value == 0.0f
 *
 * The input and output image must be of type float.
 *
 * The result is normalized to the range [0, 1]. To get the distance in pixel
 * the result has to be multiplied by \f$\sqrt{width^2 + height^2}\f$.
 *
 * It is based on Rong, G., & Tan, T. (2006). Jump flooding in GPU with
 * applications to Voronoi diagram and distance transform. In Studies in Logical
 * Theory, American Philosophical Quarterly Monograph 2, 109–116.
 * http://doi.org/10.1145/1111411.1111431
 */
class DistanceTransform
{
 public:
  DistanceTransform(std::shared_ptr<CudaArrayProvider> inputImage,
                    std::shared_ptr<CudaArrayProvider> outputImage);
  ~DistanceTransform();

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
  cudaTextureObject_t inputTexture = 0;
  cudaSurfaceObject_t outputSurface = 0;
  void prepareInputTexture();
  void prepareOutputSurface();
  void resize();
  void runInitializeKernel();
  void runStepsKernels();
  void runFinishKernel();
};

#endif  // SRC_PLACEMENT_DISTANCE_TRANSFORM_H_
