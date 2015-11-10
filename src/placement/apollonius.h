#ifndef SRC_PLACEMENT_APOLLONIUS_H_

#define SRC_PLACEMENT_APOLLONIUS_H_

#include <thrust/device_vector.h>
#include <Eigen/Core>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <deque>
#include "../utils/cuda_array_provider.h"
#include "../labelling/label.h"

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
  Apollonius(std::shared_ptr<CudaArrayProvider> distancesImage,
             std::shared_ptr<CudaArrayProvider> outputImage,
             std::vector<Eigen::Vector4f> labelPositions, int labelCount);

  void run();

  std::vector<int> calculateOrdering();

  thrust::device_vector<int> &getIds();

  std::deque<int> insertionOrder;

 private:
  std::shared_ptr<CudaArrayProvider> outputImage;
  std::shared_ptr<CudaArrayProvider> distancesImage;
  thrust::device_vector<float4> seedBuffer;
  thrust::device_vector<int> computeVector;
  thrust::device_vector<int> seedIds;
  thrust::device_vector<int> seedIndices;
  thrust::device_vector<int> orderedIndices;
  int labelCount;

  dim3 dimBlock;
  dim3 dimGrid;

  int imageSize;
  int pixelCount;

  std::set<int> extractedIndices;
  // <pixel index, label index>
  std::map<int, int> pixelIndexToLabelId;

  void extractUniqueBoundaryIndices();
  void updateLabelSeeds();

  cudaSurfaceObject_t outputSurface;
  cudaTextureObject_t distancesTexture;

  void resize();
  void runSeedKernel();
  void runStepsKernels();
  void runGatherKernel();
};

#endif  // SRC_PLACEMENT_APOLLONIUS_H_
