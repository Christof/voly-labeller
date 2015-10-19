#ifndef SRC_PLACEMENT_APOLLONIUS_H_

#define SRC_PLACEMENT_APOLLONIUS_H_

#include <thrust/device_vector.h>
#include <Eigen/Core>
#include <memory>
#include "../utils/cuda_array_provider.h"
#include "../labelling/label.h"

#define MAX_LABELS 256

/**
 * \brief Calculates the apollonius graph for the given input image,
 * seed values for labels and distances to determine the label insertion
 * order.
 *
 * It is based on Rong, G., & Tan, T. (2006). Jump flooding in GPU with
 * applications to Voronoi diagram and distance transform. In Studies in Logical
 * Theory, American Philosophical Quarterly Monograph 2, 109–116.
 * http://doi.org/10.1145/1111411.1111431
 *
 * The seedBuffer contains the following data per element:
 * - x: Id of the label
 * - y: x component of the anchor in window coordinates
 * - z: y component of the anchor in window coordinates
 * - z: is not used
 */
class Apollonius
{
 public:
  Apollonius(std::shared_ptr<CudaArrayProvider> inputImage,
             thrust::device_vector<float4> &seedBuffer,
             thrust::device_vector<float> &distances, int labelCount);

  void run();

  thrust::device_vector<int> &getIds();

  static thrust::device_vector<float4>
  createSeedBufferFromLabels(std::vector<Label> labels,
                             Eigen::Matrix4f viewProjection,
                             Eigen::Vector2i size);

 private:
  std::shared_ptr<CudaArrayProvider> inputImage;
  thrust::device_vector<float4> &seedBuffer;
  thrust::device_vector<float> &distances;
  thrust::device_vector<int> computeVector;
  thrust::device_vector<int> seedIds;
  thrust::device_vector<int> seedIndices;
  int labelCount;

  dim3 dimBlock;
  dim3 dimGrid;

  int imageSize;
  int pixelCount;

  cudaSurfaceObject_t outputSurface;

  void resize();
  void runSeedKernel();
  void runStepsKernels();
  void runGatherKernel();
};

#endif  // SRC_PLACEMENT_APOLLONIUS_H_
