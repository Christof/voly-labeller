#ifndef SRC_PLACEMENT_INTEGRAL_COSTS_WEIGHTS_H_

#define SRC_PLACEMENT_INTEGRAL_COSTS_WEIGHTS_H_

namespace Placement
{

/**
 * \brief Weights for integral costs.
 */
struct IntegralCostsWeights
{
  float occlusion = 1.0f;
  float saliency = 1e-3f;
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_INTEGRAL_COSTS_WEIGHTS_H_
