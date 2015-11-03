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
    std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper)
{
  qCInfo(plChan) << "Initialize";
  if (!occupancySummedAreaTable.get())
    occupancySummedAreaTable =
        std::make_shared<SummedAreaTable>(occupancyTextureMapper);
  this->distanceTransformTextureMapper = distanceTransformTextureMapper;
}

void Labeller::cleanup()
{
  occupancySummedAreaTable.reset();
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

  auto seedBuffer = Apollonius::createSeedBufferFromLabels(
      labels->getLabels(), frameData.viewProjection,
      Eigen::Vector2i(distanceTransformTextureMapper->getWidth(),
                      distanceTransformTextureMapper->getHeight()));
  Apollonius apollonius(distanceTransformTextureMapper,
                        distanceTransformTextureMapper, seedBuffer,
                        labels->count());
  apollonius.run();
  setInsertionOrder(apollonius.getHostIds());

  occupancySummedAreaTable->runKernel();

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  // TODO(SIR): iterate through labels specific order according to apollonius.
  for (auto id : insertionOrder)
  {
    auto label = labels->getById(id);
    auto anchor2D = frameData.project(label.anchorPosition);
    float x = (anchor2D.x() * 0.5f + 0.5f) * width;
    float y = (anchor2D.y() * 0.5f + 0.5f) * height;

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
