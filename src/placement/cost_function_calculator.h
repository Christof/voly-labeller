#ifndef SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#define SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <tuple>
#include "../utils/cuda_array_provider.h"
#include "./placement.h"

#if _WIN32
#pragma warning(disable : 4267)
#endif

namespace Placement
{

/**
 * \brief Weights for cost function terms
 */
struct CostFunctionWeights
{
  float labelShadowConstraint = 1e3f;
  float integralCosts = 5e1f;
  float distanceToAnchor = 0.8e-2f;
  float distanceToOldPosition = 0;  // 0.5e-2f;
  float favorHorizontalOrVerticalLines = 1.0f;
  float connectorShadowConstraint = 1e2f;
  float anchorConstraint = 1e5f;
};

/**
 * \brief Result of a cost function calculation
 *
 * This consists of:
 * - The position of the label
 * - The cost associated with this position (which is the minimum)
 */
struct CostFunctionResult
{
  Eigen::Vector2i position;
  float cost;
};

/**
 * \brief Calculates cost function for each pixel and returns minimum with
 * corresponding position
 *
 * The cost function consists of the following terms:
 * - integral costs of the area under the label (determined using the
 *   summed area table of the sum of the occlusion and saliency)
 * - distance between the label position to the anchor
 * - how aligned the connector line would be the horizontal or vertical axis
 * - distance between the old and new label position
 *
 * The cost function has an early out if the corresponding pixel violates the
 * constraints.
 *
 * #resize and #setTextureSize must be called before the costs can be evaluated.
 */
class CostFunctionCalculator
{
 public:
  explicit CostFunctionCalculator(
      std::shared_ptr<CudaArrayProvider> constraintImage);

  void resize(int width, int height);
  void setTextureSize(int width, int height);

  CostFunctionResult
  calculateForLabel(const thrust::device_vector<float> &integralCosts,
                    int labelId, float anchorX, float anchorY,
                    int labelWidthInPixel, int labelHeightInPixel,
                    bool ignoreOldPosition, int oldPositionX, int oldPositionY);

  CostFunctionWeights weights;

 private:
  int width;
  int height;

  int textureWidth;
  int textureHeight;

  std::shared_ptr<CudaArrayProvider> constraintImage;
  cudaTextureObject_t constraints = 0;

  void createTextureObject();
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_
