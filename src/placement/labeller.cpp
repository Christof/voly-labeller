#include "./labeller.h"
#include <Eigen/Geometry>
#include <QLoggingCategory>
#include <vector>
#include <map>
#include "../utils/cuda_array_provider.h"
#include "./summed_area_table.h"
#include "./apollonius.h"
#include "./occupancy_updater.h"
#include "./constraint_updater.h"

namespace Placement
{

QLoggingCategory plChan("Placement.Labeller");

Labeller::Labeller(std::shared_ptr<Labels> labels) : labels(labels)
{
}

void Labeller::initialize(
    std::shared_ptr<CudaArrayProvider> occupancyTextureMapper,
    std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
    std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper,
    std::shared_ptr<ConstraintUpdater> constraintUpdater)
{
  qCInfo(plChan) << "Initialize";
  if (!occupancySummedAreaTable.get())
    occupancySummedAreaTable =
        std::make_shared<SummedAreaTable>(occupancyTextureMapper);
  occupancyUpdater = std::make_shared<OccupancyUpdater>(occupancyTextureMapper);

  this->distanceTransformTextureMapper = distanceTransformTextureMapper;
  this->apolloniusTextureMapper = apolloniusTextureMapper;
  this->constraintUpdater = constraintUpdater;

  costFunctionCalculator.setTextureSize(occupancyTextureMapper->getWidth(),
                                        occupancyTextureMapper->getHeight());
}

void Labeller::cleanup()
{
  occupancySummedAreaTable.reset();
  apolloniusTextureMapper.reset();
  occupancyUpdater.reset();
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
  std::vector<Eigen::Vector4f> labelsSeed =
      createLabelSeeds(size, frameData.viewProjection);

  Apollonius apollonius(distanceTransformTextureMapper, apolloniusTextureMapper,
                        labelsSeed, labels->count());
  apollonius.run();
  setInsertionOrder(apollonius.calculateOrdering());

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  // TODO(SIR): iterate through labels specific order according to apollonius.
  for (size_t i = 0; i < insertionOrder.size(); ++i)
  {
    int id = insertionOrder[i];
    occupancySummedAreaTable->runKernel();

    auto label = labels->getById(id);
    auto anchor2D = frameData.project(label.anchorPosition);
    float x = (anchor2D.x() * 0.5f + 0.5f) * width;
    float y = (anchor2D.y() * 0.5f + 0.5f) * height;

    std::cout << "x " << int(x) << " y " << int(y) << std::endl;
    Eigen::Vector2i labelSizeForBuffer =
        label.size.cast<int>().cwiseProduct(size).cwiseQuotient(
            Eigen::Vector2i(width, height));

    auto newPosition = costFunctionCalculator.calculateForLabel(
        occupancySummedAreaTable->getResults(), label.id, x, y, label.size.x(),
        label.size.y());

    float newXPosition = std::get<0>(newPosition);
    float newYPosition = std::get<1>(newPosition);

    occupancyUpdater->addLabel(newXPosition, newYPosition,
                               labelSizeForBuffer.x(), labelSizeForBuffer.y());

    if (i + 1 < insertionOrder.size())
    {
      int nextId = insertionOrder[i + 1];
      auto nextLabel = labels->getById(nextId);
      auto nextAnchor2D = frameData.project(nextLabel.anchorPosition);
      Eigen::Vector2i nextLabelSizeForBuffer =
          nextLabel.size.cast<int>().cwiseProduct(size).cwiseQuotient(
              Eigen::Vector2i(width, height));
      constraintUpdater->addLabel(
          nextAnchor2D.head<2>().cast<int>(), nextLabelSizeForBuffer,
          anchor2D.head<2>().cast<int>(),
          Eigen::Vector2i(newXPosition, newYPosition), labelSizeForBuffer);
    }

    float newXNDC = (newXPosition / size.x() - 0.5f) * 2.0f;
    float newYNDC = (newYPosition / size.y() - 0.5f) * 2.0f;
    Eigen::Vector4f reprojected =
        inverseViewProjection *
        Eigen::Vector4f(newXNDC, newYNDC, anchor2D.z(), 1);
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

std::vector<Eigen::Vector4f>
Labeller::createLabelSeeds(Eigen::Vector2i size, Eigen::Matrix4f viewProjection)
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
