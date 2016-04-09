#ifndef SRC_PLACEMENT_OCCLUSION_CALCULATOR_H_

#define SRC_PLACEMENT_OCCLUSION_CALCULATOR_H_

#include <memory>
#include <vector>

class TextureMapperManager;

namespace Placement
{

class Occlusion;

/**
 * \brief Calculates the occlusion for a given layer
 *
 * The calls to #calculateFor must be in ascending layer index order.
 */
class OcclusionCalculator
{
 public:
  explicit OcclusionCalculator(int layerCount);
  void initialize(std::shared_ptr<TextureMapperManager> textureMapperManager);

  void calculateFor(int layerIndex);

 private:
  int layerCount;
  std::vector<std::shared_ptr<Occlusion>> occlusions;
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_OCCLUSION_CALCULATOR_H_
