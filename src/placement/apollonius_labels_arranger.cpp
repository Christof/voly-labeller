#include "./apollonius_labels_arranger.h"
#include <vector>
#include "../utils/cuda_array_provider.h"
#include "./apollonius.h"
#include "./distance_transform.h"

namespace Placement
{

void ApolloniusLabelsArranger::initialize(
    std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
    std::shared_ptr<CudaArrayProvider> occlusionTextureMapper,
    std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper)
{
  this->distanceTransformTextureMapper = distanceTransformTextureMapper;
  this->occlusionTextureMapper = occlusionTextureMapper;
  this->apolloniusTextureMapper = apolloniusTextureMapper;
}

std::vector<Label> ApolloniusLabelsArranger::getArrangement(
    const LabellerFrameData &frameData, std::shared_ptr<LabelsContainer> labels)
{
  DistanceTransform distanceTransform(occlusionTextureMapper,
                                      distanceTransformTextureMapper);
  distanceTransform.run();

  if (labels->count() == 1)
    return std::vector<Label>{ labels->getLabels()[0] };

  Eigen::Vector2i bufferSize(distanceTransformTextureMapper->getWidth(),
                             distanceTransformTextureMapper->getHeight());
  std::vector<Eigen::Vector4f> labelsSeed =
      createLabelSeeds(bufferSize, frameData.viewProjection, labels);

  Apollonius apollonius(distanceTransformTextureMapper, apolloniusTextureMapper,
                        labelsSeed, labels->count());
  apollonius.run();

  std::vector<int> apolloniusOrder = apollonius.calculateOrdering();
  std::vector<Label> result;
  for (int id : apolloniusOrder)
  {
    result.push_back(labels->getById(id));
  }

  return result;
}

void ApolloniusLabelsArranger::cleanup()
{
  distanceTransformTextureMapper.reset();
  occlusionTextureMapper.reset();
  apolloniusTextureMapper.reset();
}

std::vector<Eigen::Vector4f> ApolloniusLabelsArranger::createLabelSeeds(
    Eigen::Vector2i size, Eigen::Matrix4f viewProjection,
    std::shared_ptr<LabelsContainer> labels)
{
  std::vector<Eigen::Vector4f> result;
  for (auto &label : labels->getLabels())
  {
    Eigen::Vector4f pos =
        viewProjection * Eigen::Vector4f(label.anchorPosition.x(),
                                         label.anchorPosition.y(),
                                         label.anchorPosition.z(), 1);
    float x = (pos.x() / pos.w() * 0.5f + 0.5f) * size.x();
    float y = (pos.y() / pos.w() * 0.5f + 0.5f) * size.y();
    result.push_back(Eigen::Vector4f(label.id, x, y, 1));
  }

  return result;
}

}  // namespace Placement
