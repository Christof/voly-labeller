#include "./labeller.h"
#include <Eigen/Geometry>
#include <vector>
#include <QLoggingCategory>
#include "../utils/cuda_array_provider.h"
#include "./summed_area_table.h"
#include "./apollonius.h"

namespace Placement
{

QLoggingCategory plChan("Placement.Labeller");

Labeller::Labeller(std::shared_ptr<Labels> labels) : labels(labels)
{
}

void Labeller::initialize(
    std::shared_ptr<CudaArrayProvider> occupancyTextureMapper,
    std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
    std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper)
{
  qCInfo(plChan) << "Initialize";
  if (!occupancySummedAreaTable.get())
    occupancySummedAreaTable =
        std::make_shared<SummedAreaTable>(occupancyTextureMapper);

  this->distanceTransformTextureMapper = distanceTransformTextureMapper;
  this->apolloniusTextureMapper = apolloniusTextureMapper;
}

void Labeller::cleanup()
{
  occupancySummedAreaTable.reset();
  apolloniusTextureMapper.reset();
}

void Labeller::setInsertionOrder(std::vector<int> ids)
{
  insertionOrder = ids;
}

std::map<int, Eigen::Vector3f>
Labeller::update(const LabellerFrameData &frameData)
{
  newPositions.clear();
  if (!occupancySummedAreaTable.get())
    return newPositions;

  Eigen::Vector2i size(distanceTransformTextureMapper->getWidth(),
                       distanceTransformTextureMapper->getHeight());
  std::vector<Eigen::Vector4f> labelsSeed;
  for (auto &label : labels->getLabels())
  {
    Eigen::Vector4f pos =
        frameData.viewProjection * Eigen::Vector4f(label.anchorPosition.x(),
                                                   label.anchorPosition.y(),
                                                   label.anchorPosition.z(), 1);
    float x = (pos.x() / pos.w() * 0.5f + 0.5f) * size.x();
    float y = (pos.y() / pos.w() * 0.5f + 0.5f) * size.y();
    labelsSeed.push_back(Eigen::Vector4f(label.id, x, y, 1));
  }

  Apollonius apollonius(distanceTransformTextureMapper, apolloniusTextureMapper,
                        labelsSeed, labels->count());
  apollonius.run();
  setInsertionOrder(apollonius.calculateOrdering());

  occupancySummedAreaTable->runKernel();

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  // TODO(SIR): iterate through labels specific order according to apollonius.
  for (auto id : insertionOrder)
  {
    auto label = labels->getById(id);
    auto anchor2D = frameData.project(label.anchorPosition);
    float x = (anchor2D.x() * 0.5f + 0.5f) * width;
    float y = (anchor2D.y() * 0.5f + 0.5f) * height;

    std::cout << "x " << int(x) << " y " << int(y) << std::endl;

    auto newPosition = costFunctionCalculator.calculateForLabel(
        occupancySummedAreaTable->getResults(), label.id, x, y,
        label.size.x() * width, label.size.y() * height);

    // TODO(SIR): update occupancy and recalculate SAT or incorporate labels
    // somehow directly in occupancy calculation in CostFunctionCalculator.

    float newX = (std::get<0>(newPosition) / width - 0.5f) * 2.0f;
    float newY = (std::get<1>(newPosition) / height - 0.5f) * 2.0f;
    Eigen::Vector4f reprojected =
        inverseViewProjection * Eigen::Vector4f(newX, newY, anchor2D.z(), 1);
    reprojected /= reprojected.w();

    newPositions[label.id] = toVector3f(reprojected);
  }

  return newPositions;
}

void Labeller::resize(int width, int height)
{
  this->width = width;
  this->height = height;

  costFunctionCalculator.resize(width, height);
}

std::map<int, Eigen::Vector3f> Labeller::getLastPlacementResult()
{
  return newPositions;
}

}  // namespace Placement
