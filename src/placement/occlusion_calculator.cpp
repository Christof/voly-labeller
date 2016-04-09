#include "./occlusion_calculator.h"
#include <vector>
#include "../texture_mapper_manager.h"
#include "./cuda_texture_mapper.h"
#include "./occlusion.h"

namespace Placement
{

OcclusionCalculator::OcclusionCalculator(int layerCount)
  : layerCount(layerCount)
{
}

void OcclusionCalculator::initialize(
    std::shared_ptr<TextureMapperManager> textureMapperManager)
{
  for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
  {
    occlusions.push_back(std::make_shared<Occlusion>(
          textureMapperManager->getColorTextureMapper(layerIndex),
          textureMapperManager->getOcclusionTextureMapper()));
  }
}

void OcclusionCalculator::calculateFor(int layerIndex)
{
  if (layerIndex == 0)
  {
    occlusions[0]->calculateOcclusion();
    return;
  }

  occlusions[layerIndex]->addOcclusion();
}

}  // namespace Placement
